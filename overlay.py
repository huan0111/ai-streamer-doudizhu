import ctypes
import tkinter as tk


def overlay_geometry(game_rect, screen_w, screen_h, width=600, height=330):
    left, top, right, bottom = game_rect
    candidates = [
        (right + 8, top),
        (left, bottom + 8),
        (left - width - 8, top),
        (left, top - height - 8),
    ]
    for x, y in candidates:
        on_screen = x >= 0 and y >= 0 and x + width <= screen_w and y + height <= screen_h
        overlaps_game = x < right and x + width > left and y < bottom and y + height > top
        if on_screen and not overlaps_game:
            return x, y
    return None


class Overlay:
    def __init__(self, width=600, height=330, x=650, y=35, on_takeover=None):
        self.width = width
        self.height = height
        self.on_takeover = on_takeover
        self.root = tk.Tk()
        self.root.title('AI 斗地主 · 实时口播')
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        self.root.attributes('-topmost', True)
        self.root.configure(bg='#0f1419')
        self.root.protocol('WM_DELETE_WINDOW', lambda: None)
        header = tk.Label(
            self.root, text='实时口播（人设润色前） · 下方为决策日志',
            bg='#16212b', fg='#7fd1ff', font=('Microsoft YaHei', 10, 'bold'), anchor='w', padx=8, pady=4,
        )
        header.pack(fill='x')
        self.speech = tk.Label(
            self.root, text='等待识别游戏画面…', bg='#101c18', fg='#7ee787',
            font=('Microsoft YaHei', 15, 'bold'), anchor='w', justify='left',
            wraplength=width - 28, padx=10, pady=10, height=3,
        )
        self.speech.pack(fill='x', padx=8, pady=6)
        self.takeover_button = tk.Button(
            self.root, text='人工接管（Ctrl+Alt+P）', command=lambda: self.on_takeover and self.on_takeover(),
            bg='#b03a2e', fg='white', font=('Microsoft YaHei', 10, 'bold'), bd=0, padx=8, pady=5,
        )
        self.takeover_button.pack(fill='x', padx=8, pady=3)
        self.text = tk.Text(
            self.root, bg='#0f1419', fg='#e6edf3', font=('Microsoft YaHei', 9),
            wrap='word', bd=0, padx=8, pady=6, insertbackground='#e6edf3',
        )
        self.text.pack(fill='both', expand=True)
        self.text.configure(state='disabled')
        self.text.tag_configure('decision', foreground='#7ee787')
        self.text.tag_configure('error', foreground='#ff7b72')
        self.text.tag_configure('warn', foreground='#e3b341')
        self.text.tag_configure('dim', foreground='#8b949e')
        self.root.update()
        self.raise_topmost()

    def raise_topmost(self):
        try:
            user32 = ctypes.windll.user32
            hwnd = self.root.winfo_id()
            parent = user32.GetParent(hwnd)
            target = parent or hwnd
            user32.SetWindowPos(target, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)
        except Exception:
            pass

    def place_outside(self, game_rect):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = self.width
        space = screen_w - (game_rect[2] + 8)
        if space < width and space >= 340:
            width = space
        position = overlay_geometry(game_rect, screen_w, screen_h, width, self.height)
        if position is None:
            self.root.withdraw()
            return
        x, y = position
        self.root.deiconify()
        self.root.geometry(f'{width}x{self.height}+{x}+{y}')
        self.speech.configure(wraplength=max(200, width - 28))
        self.root.update_idletasks()
        self.raise_topmost()

    def set_paused(self, paused):
        self.takeover_button.configure(
            text='恢复 AI（Ctrl+Alt+P）' if paused else '人工接管（Ctrl+Alt+P）',
            bg='#246b45' if paused else '#b03a2e',
        )
        self.speech.configure(
            text='人工接管中：AI 已停止点击，按 Ctrl+Alt+P 恢复' if paused else 'AI 自动操作中',
            fg='#ffb86c' if paused else '#7ee787',
        )
        self.update()

    def show_speech(self, message, event_kind=None):
        correcting = event_kind == 'execution_result'
        self.speech.configure(
            text=message,
            fg='#ffb86c' if correcting else '#7ee787',
            bg='#2a2116' if correcting else '#101c18',
        )
        try:
            self.raise_topmost()
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass

    def push(self, message, tag=None):
        self.text.configure(state='normal')
        self.text.insert('end', message + '\n', tag or ())
        self.text.see('end')
        self.text.configure(state='disabled')
        try:
            self.raise_topmost()
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass

    def update(self):
        try:
            self.raise_topmost()
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass
