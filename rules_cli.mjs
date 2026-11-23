import fs from 'node:fs'
import { performance } from 'node:perf_hooks'
import {
  cardFromId,
  classifyMove,
  enumerateBeats,
  moveKey,
} from './vendor/dsh-doudizhu/src/rules.js'

const suits = ['S', 'H', 'D', 'C']

function rankOf(cardId) {
  return cardId === 'BJ' || cardId === 'LJ' ? cardId : cardId.slice(1)
}

function mapRanks(ranks) {
  const rankCounts = new Map()
  const indexById = new Map()
  const cardIds = ranks.map((rank, index) => {
    if (rank === 'BJ' || rank === 'LJ') {
      if (rankCounts.has(rank)) throw new Error(`duplicate joker: ${rank}`)
      rankCounts.set(rank, 1)
      indexById.set(rank, index)
      return rank
    }
    if (!/^(10|[2-9AJQK])$/.test(rank)) throw new Error(`unknown rank: ${rank}`)
    const count = rankCounts.get(rank) || 0
    if (count >= 4) throw new Error(`too many cards for rank: ${rank}`)
    rankCounts.set(rank, count + 1)
    const cardId = `${suits[count]}${rank}`
    indexById.set(cardId, index)
    return cardId
  })
  return { cardIds, indexById }
}

function selectedIds(ranks, hand) {
  const used = new Set()
  return ranks.map((rank) => {
    const index = hand.cardIds.findIndex((cardId, candidateIndex) => {
      return !used.has(candidateIndex) && rankOf(cardId) === rank
    })
    if (index < 0) throw new Error(`selected rank is not in hand: ${rank}`)
    used.add(index)
    return hand.cardIds[index]
  })
}

function main() {
  const [inputPath, outputPath] = process.argv.slice(2)
  if (!inputPath || !outputPath) {
    throw new Error('usage: node rules_cli.mjs <input.json> <output.json>')
  }
  const request = JSON.parse(fs.readFileSync(inputPath, 'utf8'))
  const started = performance.now()
  const hand = mapRanks(request.my_hand)
  const targetIds = request.last_play?.ranks?.length
    ? mapRanks(request.last_play.ranks).cardIds
    : null
  const target = targetIds ? classifyMove(targetIds) : null
  if (targetIds && !target) throw new Error('last_play is not a legal move')
  const candidates = enumerateBeats(hand.cardIds, target, {
    limit: request.limit || 400,
  }).map((candidate, index) => ({
    id: `c${String(index).padStart(3, '0')}`,
    move: candidate.move,
    move_key: moveKey(candidate.move),
    ranks: candidate.cardIds.map(rankOf),
    card_ids: candidate.cardIds,
    source_indices: candidate.cardIds.map((cardId) => hand.indexById.get(cardId)),
  }))
  if (target) {
    candidates.push({ id: 'pass', move: { type: 'pass' }, ranks: [], card_ids: [], source_indices: [] })
  }
  const selectedCardIds = request.selected_indices?.length
    ? request.selected_indices.map((index) => {
      if (!Number.isInteger(index) || index < 0 || index >= hand.cardIds.length) {
        throw new Error(`invalid selected index: ${index}`)
      }
      return hand.cardIds[index]
    })
    : request.selected_ranks?.length
      ? selectedIds(request.selected_ranks, hand)
      : []
  const selectedMove = selectedCardIds.length ? classifyMove(selectedCardIds) : null
  const selectedKey = moveKey(selectedMove)
  const output = {
    hand_card_ids: hand.cardIds,
    target,
    can_pass: Boolean(target),
    fast_path_pass: Boolean(target && candidates.length === 1),
    selected_move: selectedMove,
    selected_card_ids: selectedCardIds,
    selected_indices: request.selected_indices || [],
    selected_candidate_id: candidates.find((candidate) => candidate.move_key === selectedKey)?.id || null,
    candidates,
    elapsed_ms: Number((performance.now() - started).toFixed(3)),
  }
  fs.writeFileSync(outputPath, `${JSON.stringify(output, null, 2)}\n`)
}

main()
