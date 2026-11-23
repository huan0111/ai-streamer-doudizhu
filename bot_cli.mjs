import fs from 'node:fs'
import { chooseAction, bidPower } from './vendor/dsh-doudizhu/src/bot.js'
import { PHASES } from './vendor/dsh-doudizhu/src/game.js'
import { beatableFromOutstanding, cardFromId, classifyMove, enumerateBeats, enumerateMoves, moveKey, movesToEmpty, removeCards } from './vendor/dsh-doudizhu/src/rules.js'
import { outstandingFor } from './vendor/dsh-doudizhu/src/bot.js'

const suits = ['S', 'H', 'D', 'C']

function mapRanks(ranks = []) {
  const counts = new Map()
  const ids = []
  const indexById = new Map()
  for (const rank of ranks) {
    if (rank === 'BJ' || rank === 'LJ') {
      ids.push(rank)
      indexById.set(rank, ids.length - 1)
      continue
    }
    const count = counts.get(rank) || 0
    if (count >= 4) throw new Error(`too many cards for rank: ${rank}`)
    counts.set(rank, count + 1)
    const id = `${suits[count]}${rank}`
    ids.push(id)
    indexById.set(id, ids.length - 1)
  }
  return { ids, indexById }
}

function main() {
  const [inputPath, outputPath] = process.argv.slice(2)
  if (!inputPath || !outputPath) throw new Error('usage: node bot_cli.mjs <input.json> <output.json>')
  const request = JSON.parse(fs.readFileSync(inputPath, 'utf8'))
  const hand = mapRanks(request.hand)
  const role = request.role === 'landlord' ? 'landlord' : 'farmer'
  const difficulty = request.difficulty || 'hell'

  if (request.phase === 'bidding') {
    const power = bidPower(hand.ids)
    const view = {
      difficulty, seat: 0, phase: PHASES.BIDDING, role, hand: hand.ids,
      handCounts: [{ seat: 0, role, count: hand.ids.length }],
      playedIds: [], landlordSeat: null, dealerSeat: 0, toAct: null,
      target: null, targetSeat: null, passes: 0, canPass: false,
      bidTurnSeat: 0, highestBid: { score: request.highestBid || 0 },
    }
    const action = chooseAction(view, { rand: () => 0.5 })
    fs.writeFileSync(outputPath, `${JSON.stringify({ power, action, phase: 'bidding' }, null, 2)}\n`)
    return
  }

  const target = request.target?.length ? mapRanks(request.target) : null
  const targetMove = target ? classifyMove(target.ids) : null
  if (target && !targetMove) throw new Error('target is not a legal move')
  const played = mapRanks(request.played || [])
  const counts = request.handCounts || {}
  const leftRole = role === 'landlord' ? 'farmer' : (counts.leftLandlord ? 'landlord' : 'farmer')
  const rightRole = role === 'landlord' ? 'farmer' : (counts.rightLandlord ? 'landlord' : 'farmer')
  const view = {
    difficulty, seat: 0, phase: PHASES.PLAYING, role,
    hand: hand.ids,
    handCounts: [
      { seat: 0, role, count: hand.ids.length },
      { seat: 1, role: leftRole, count: Number(counts.left ?? 17) },
      { seat: 2, role: rightRole, count: Number(counts.right ?? 17) },
    ],
    playedIds: played.ids,
    landlordSeat: role === 'landlord' ? 0 : (leftRole === 'landlord' ? 1 : 2),
    dealerSeat: 0,
    toAct: 0,
    target: targetMove,
    targetSeat: request.targetSeat === 'left' ? 1 : request.targetSeat === 'right' ? 2 : null,
    passes: Number(request.passes || 0),
    canPass: Boolean(request.canPass),
    bidTurnSeat: null,
    highestBid: { score: 0 },
  }
  const action = chooseAction(view, { rand: () => 0.5 })
  const result = { phase: 'playing', role, action }

  if (request.mode === 'candidates') {
    const moves = targetMove ? enumerateBeats(hand.ids, targetMove) : enumerateMoves(hand.ids)
    const outstanding = outstandingFor(view)
    const countRanks = (ids) => {
      const map = new Map()
      for (const id of ids) {
        if (id === 'BJ' || id === 'LJ') continue
        const rank = id.slice(1)
        map.set(rank, (map.get(rank) || 0) + 1)
      }
      return map
    }
    const handCounts = countRanks(hand.ids)
    const rows = moves.map((move, index) => {
      const remaining = removeCards(hand.ids, move.cardIds)
      const movesAfter = movesToEmpty(remaining)
      const ranks = move.cardIds.map((id) => (id === 'BJ' || id === 'LJ' ? id : id.slice(1)))
      const usedCounts = countRanks(move.cardIds)
      const breaksBomb = [...usedCounts.entries()].some(([rank, used]) => (handCounts.get(rank) || 0) >= 4 && used < 4)
      const breaksTriple = [...usedCounts.entries()].some(([rank, used]) => (handCounts.get(rank) || 0) === 3 && used < 3)
      const usesBomb = Boolean(move.move.isBomb || move.move.isRocket) || [...usedCounts.entries()].some(([rank, used]) => used >= 4)
      const usesHigh = move.cardIds.some((id) => (cardFromId(id)?.value ?? 0) >= 15)
      const wins = remaining.length === 0
      const unanswerable = !beatableFromOutstanding(move.move, outstanding)
      const indices = move.cardIds.map((id) => hand.indexById.get(id))
      const reasons = []
      if (wins) reasons.push('这一手直接走完')
      if (unanswerable && !wins) reasons.push('这一手当前无人能压')
      if (usesBomb && !wins) reasons.push('动用炸弹/王炸')
      if (usesHigh) reasons.push('动用 2 或王')
      if (breaksBomb) reasons.push('拆了四张')
      if (breaksTriple) reasons.push('拆了三张')
      if (!wins && !reasons.length) reasons.push(`打完后约剩 ${movesAfter} 手`)
      return {
        id: `c${String(index).padStart(3, '0')}`,
        move_key: moveKey(move.move),
        type: move.move.type,
        label: move.move.label,
        ranks,
        indices,
        cards: move.cardIds,
        size: move.move.size,
        main_value: move.move.mainValue,
        moves_after: movesAfter,
        wins,
        unanswerable,
        uses_bomb: usesBomb,
        uses_high: usesHigh,
        breaks_bomb: breaksBomb,
        breaks_triple: breaksTriple,
        reasons,
        local_score: (wins ? -1000 : 0) + (unanswerable ? -2 : 0) + movesAfter * 10 + (usesBomb ? 25 : 0) + (usesHigh ? 6 : 0) + (breaksBomb ? 8 : 0) + (breaksTriple ? 8 : 0) - move.move.size * 0.5,
      }
    })
    rows.sort((left, right) => left.local_score - right.local_score || right.size - left.size || left.main_value - right.main_value)
    const limit = Number(request.candidateLimit || 6)
    result.candidates = rows.slice(0, limit)
    if (view.canPass && targetMove && (action?.type === 'pass' || rows.length === 0)) {
      result.candidates.push({ id: 'pass', type: 'pass', label: '不出', ranks: [], indices: [], cards: [], size: 0, main_value: 0, reasons: ['放弃这一手'], local_score: 50 })
    }
    result.recommended = result.candidates[0]?.id ?? null
    result.recommended_source = 'local_score'
    if (action && action.type === 'play') {
      const key = moveKey(classifyMove(action.cards))
      const actionCards = [...action.cards].sort().join(',')
      const match = rows.find((item) => item.move_key === key && item.type !== 'pass'
        && [...item.cards].sort().join(',') === actionCards)
      if (match) {
        if (!result.candidates.some((item) => item.id === match.id)) {
          if (limit > 0) result.candidates[limit - 1] = match
          else result.candidates.push(match)
        }
        result.recommended = match.id
        result.recommended_source = 'bot.js'
      }
    } else if (action && action.type === 'pass') {
      const match = result.candidates.find((item) => item.id === 'pass')
      if (match) {
        result.recommended = 'pass'
        result.recommended_source = 'bot.js'
      }
    }
    result.outstanding = Object.fromEntries([...outstanding.entries()].map(([value, count]) => [String(value), count]))
  }
  if (action?.type === 'play') {
    const indices = action.cards.map((id) => hand.indexById.get(id))
    const ranks = action.cards.map((id) => (id === 'BJ' || id === 'LJ' ? id : id.slice(1)))
    const move = classifyMove(action.cards)
    result.action = { type: 'play', cards: action.cards, ranks, indices, move }
  }
  fs.writeFileSync(outputPath, `${JSON.stringify(result, null, 2)}\n`)
}

main()
