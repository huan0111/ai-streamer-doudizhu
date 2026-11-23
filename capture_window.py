import argparse
import ctypes
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageGrab


class RECT(ctypes.Structure):
    _fields_ = [('left', ctypes.c_long), ('top', ctypes.c_long), ('right', ctypes.c_long), ('bottom', ctypes.c_long)]


class POINT(ctypes.Structure):
    _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]


def window_rect(user32, hwnd):
    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError('GetWindowRect failed')
    return rect


def client_bbox(user32, hwnd):
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError('GetClientRect failed')
    point = POINT(rect.left, rect.top)
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        raise RuntimeError('ClientToScreen failed')
    return point.x, point.y, point.x + rect.right - rect.left, point.y + rect.bottom - rect.top


def find_window(title_part, class_name='UnityWndClass'):
    user32 = ctypes.windll.user32
    matches = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if not length:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if buffer.value == title_part:
            class_buffer = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, class_buffer, 256)
            if class_name and class_buffer.value != class_name:
                return True
            rect = window_rect(user32, hwnd)
            matches.append((hwnd, buffer.value, rect))
        return True

    user32.EnumWindows(callback_type(callback), 0)
    if not matches:
        raise RuntimeError(f'no visible window contains title: {title_part}')
    return user32, matches[0]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ('biSize', ctypes.c_uint32), ('biWidth', ctypes.c_int32), ('biHeight', ctypes.c_int32),
        ('biPlanes', ctypes.c_uint16), ('biBitCount', ctypes.c_uint16), ('biCompression', ctypes.c_uint32),
        ('biSizeImage', ctypes.c_uint32), ('biXPelsPerMeter', ctypes.c_int32), ('biYPelsPerMeter', ctypes.c_int32),
        ('biClrUsed', ctypes.c_uint32), ('biClrImportant', ctypes.c_uint32),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [('bmiHeader', BITMAPINFOHEADER), ('bmiColors', ctypes.c_uint32 * 3)]


def print_window_image(hwnd, width, height):
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    hdc = user32.GetDC(hwnd)
    if not hdc:
        return None
    memory = gdi32.CreateCompatibleDC(hdc)
    bitmap = gdi32.CreateCompatibleBitmap(hdc, width, height)
    previous = gdi32.SelectObject(memory, bitmap)
    ok = user32.PrintWindow(hwnd, memory, 3)
    info = BITMAPINFO()
    info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    info.bmiHeader.biWidth = width
    info.bmiHeader.biHeight = -height
    info.bmiHeader.biPlanes = 1
    info.bmiHeader.biBitCount = 32
    buffer = ctypes.create_string_buffer(width * height * 4)
    got = gdi32.GetDIBits(memory, bitmap, 0, height, buffer, ctypes.byref(info), 0)
    gdi32.SelectObject(memory, previous)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(memory)
    user32.ReleaseDC(hwnd, hdc)
    if not ok or got != height:
        return None
    array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    return Image.fromarray(array[:, :, ::-1])


def save_image(path, bbox):
    image = ImageGrab.grab(bbox=bbox, all_screens=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format='PNG')
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', default='欢乐斗地主')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--client-output', type=Path)
    args = parser.parse_args()
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()
    user32, (hwnd, title, rect) = find_window(args.title)
    bbox = (rect.left, rect.top, rect.right, rect.bottom)
    client_box = client_bbox(user32, hwnd)
    image = save_image(args.output, bbox)
    metadata = {
        'window_title': title,
        'hwnd': hwnd,
        'window_rect': bbox,
        'client_rect': client_box,
        'image_size': list(image.size),
        'captured_at': datetime.now().astimezone().isoformat(),
        'capture_method': 'PIL.ImageGrab; no activation, resize, or click',
        'limitation': 'Foreground windows may occlude the game; inspect the image before recognition. Win32 PrintWindow was attempted and rejected by this Unity window.',
    }
    if args.client_output:
        client_image = save_image(args.client_output, client_box)
        metadata['client_image'] = str(args.client_output)
        metadata['client_image_size'] = list(client_image.size)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=True))


if __name__ == '__main__':
    main()
