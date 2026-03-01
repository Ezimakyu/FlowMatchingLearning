import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
} from 'd3-force'

import type { FlowEdge, FlowNode, LayoutMode } from './types'

type TopologicalLevelMap = Map<string, number>

function buildTopologicalLevels(nodes: FlowNode[], edges: FlowEdge[]): TopologicalLevelMap {
  const indegree = new Map<string, number>()
  const adjacency = new Map<string, string[]>()
  const level = new Map<string, number>()

  for (const node of nodes) {
    indegree.set(node.id, 0)
    adjacency.set(node.id, [])
    level.set(node.id, 0)
  }

  for (const edge of edges) {
    if (!indegree.has(edge.source) || !indegree.has(edge.target)) {
      continue
    }
    indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1)
    adjacency.get(edge.source)?.push(edge.target)
  }

  const queue = nodes
    .map((node) => node.id)
    .filter((nodeId) => (indegree.get(nodeId) ?? 0) === 0)
    .sort()

  const visited = new Set<string>()
  while (queue.length > 0) {
    const nodeId = queue.shift()
    if (!nodeId) {
      continue
    }
    visited.add(nodeId)
    const parentLevel = level.get(nodeId) ?? 0
    const children = adjacency.get(nodeId) ?? []
    for (const childId of children) {
      const previous = level.get(childId) ?? 0
      if (parentLevel + 1 > previous) {
        level.set(childId, parentLevel + 1)
      }
      const nextInDegree = (indegree.get(childId) ?? 0) - 1
      indegree.set(childId, nextInDegree)
      if (nextInDegree === 0) {
        queue.push(childId)
      }
    }
    queue.sort()
  }

  if (visited.size !== nodes.length) {
    // Graph is expected to be a DAG, but this fallback avoids hard failure.
    let fallbackLevel = Math.max(...Array.from(level.values(), (value) => value), 0) + 1
    for (const node of nodes) {
      if (visited.has(node.id)) {
        continue
      }
      level.set(node.id, fallbackLevel)
      fallbackLevel += 1
    }
  }

  return level
}

function applyTopologicalLayout(nodes: FlowNode[], edges: FlowEdge[]): FlowNode[] {
  const levels = buildTopologicalLevels(nodes, edges)
  const groups = new Map<string, FlowNode[]>()
  for (const node of nodes) {
    const groupKey = node.data.sectionId?.split('/')[0] || 'ungrouped'
    const bucket = groups.get(groupKey) ?? []
    bucket.push(node)
    groups.set(groupKey, bucket)
  }

  const sortedGroups = Array.from(groups.entries()).sort((left, right) => {
    const leftLevels = left[1].map((node) => levels.get(node.id) ?? 0)
    const rightLevels = right[1].map((node) => levels.get(node.id) ?? 0)
    const leftAvg = leftLevels.reduce((sum, value) => sum + value, 0) / leftLevels.length
    const rightAvg = rightLevels.reduce((sum, value) => sum + value, 0) / rightLevels.length
    if (leftAvg !== rightAvg) {
      return leftAvg - rightAvg
    }
    return left[0].localeCompare(right[0])
  })
  const groupGap = 480
  const arcRadiusBase = 90
  const laidOut = new Map<string, FlowNode>()

  sortedGroups.forEach(([groupKey, bucket], groupIndex) => {
    const centerX = groupIndex * groupGap
    const centerY = 0
    const sortedBucket = [...bucket].sort((left, right) => {
      const leftLevel = levels.get(left.id) ?? 0
      const rightLevel = levels.get(right.id) ?? 0
      if (leftLevel !== rightLevel) {
        return leftLevel - rightLevel
      }
      return left.data.label.localeCompare(right.data.label)
    })

    if (sortedBucket.length === 1) {
      const only = sortedBucket[0]
      laidOut.set(only.id, {
        ...only,
        position: {
          x: centerX,
          y: centerY,
        },
      })
      return
    }

    const radius = arcRadiusBase + sortedBucket.length * 12
    const startAngle = Math.PI
    const endAngle = 0
    const spread = startAngle - endAngle
    sortedBucket.forEach((node, index) => {
      const ratio = sortedBucket.length === 1 ? 0.5 : index / (sortedBucket.length - 1)
      const angle = startAngle - spread * ratio
      const levelBiasX = (levels.get(node.id) ?? 0) * 28
      laidOut.set(node.id, {
        ...node,
        position: {
          x: centerX + Math.cos(angle) * radius * 0.18 + levelBiasX,
          y: centerY + Math.sin(angle) * radius * 1.05 - radius * 0.55,
        },
      })
    })

    // Keep group key referenced to avoid dead-code elimination warnings in some configs.
    void groupKey
  })

  return nodes.map((node) => laidOut.get(node.id) ?? node)
}

interface ForceNode {
  id: string
  x: number
  y: number
}

interface ForceLink {
  source: string
  target: string
}

function applyForceLayout(nodes: FlowNode[], edges: FlowEdge[]): FlowNode[] {
  if (nodes.length === 0) {
    return []
  }
  const forceNodes: ForceNode[] = nodes.map((node, index) => ({
    id: node.id,
    x: node.position.x || (index % 8) * 80,
    y: node.position.y || Math.floor(index / 8) * 80,
  }))
  const forceLinks: ForceLink[] = edges.map((edge) => ({
    source: edge.source,
    target: edge.target,
  }))
  const simulation = forceSimulation<ForceNode>(forceNodes)
    .force(
      'link',
      forceLink<ForceNode, ForceLink>(forceLinks)
        .id((node) => node.id)
        .distance(160)
        .strength(0.22)
    )
    .force('charge', forceManyBody().strength(-430))
    .force('center', forceCenter(0, 0))
    .force('collision', forceCollide(55))

  for (let tick = 0; tick < 220; tick += 1) {
    simulation.tick()
  }
  simulation.stop()

  const positionById = new Map(
    forceNodes.map((node) => [node.id, { x: node.x || 0, y: node.y || 0 }])
  )
  return nodes.map((node) => ({
    ...node,
    position: positionById.get(node.id) ?? node.position,
  }))
}

export function applyLayoutMode(params: {
  mode: LayoutMode
  nodes: FlowNode[]
  edges: FlowEdge[]
}): FlowNode[] {
  const { mode, nodes, edges } = params
  if (mode === 'force') {
    return applyForceLayout(nodes, edges)
  }
  return applyTopologicalLayout(nodes, edges)
}
