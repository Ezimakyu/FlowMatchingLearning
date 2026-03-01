import { MarkerType } from 'reactflow'

import type {
  ConceptEdgeRecord,
  ConceptNodeRecord,
  FlowEdge,
  FlowNode,
  GraphData,
} from './types'

function asArray<T>(value: T[] | undefined | null): T[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value
}

export function parseGraphData(payload: unknown): GraphData {
  if (payload === null || typeof payload !== 'object') {
    throw new Error('Graph payload must be a JSON object.')
  }
  const candidate = payload as Partial<GraphData>
  if (!Array.isArray(candidate.nodes) || !Array.isArray(candidate.edges)) {
    throw new Error('graph_data.json must include "nodes" and "edges" arrays.')
  }
  return {
    schema_version: candidate.schema_version,
    graph_id: candidate.graph_id ?? null,
    generated_at: candidate.generated_at,
    nodes: candidate.nodes.map((node, index) => normalizeNode(node, index)),
    edges: candidate.edges.map((edge, index) => normalizeEdge(edge, index)),
    metadata: candidate.metadata ?? {},
  }
}

function normalizeNode(node: unknown, index: number): ConceptNodeRecord {
  if (node === null || typeof node !== 'object') {
    throw new Error(`Node at index ${index} is not an object.`)
  }
  const candidate = node as Partial<ConceptNodeRecord>
  if (!candidate.id || !candidate.label || !candidate.summary || !candidate.source_material) {
    throw new Error(`Node at index ${index} is missing required fields.`)
  }
  return {
    id: String(candidate.id),
    label: String(candidate.label),
    summary: String(candidate.summary),
    aliases: asArray(candidate.aliases).map(String),
    confidence: typeof candidate.confidence === 'number' ? candidate.confidence : 1,
    deep_dive: candidate.deep_dive ?? null,
    source_material: {
      doc_id: String(candidate.source_material.doc_id),
      section_id: candidate.source_material.section_id ?? null,
      chunk_ids: asArray(candidate.source_material.chunk_ids).map(String),
      page_numbers: asArray(candidate.source_material.page_numbers).map(Number),
      transcript_timestamps: asArray(candidate.source_material.transcript_timestamps).map(String),
      snippet: candidate.source_material.snippet ?? null,
    },
    metadata: candidate.metadata ?? {},
  }
}

function normalizeEdge(edge: unknown, index: number): ConceptEdgeRecord {
  if (edge === null || typeof edge !== 'object') {
    throw new Error(`Edge at index ${index} is not an object.`)
  }
  const candidate = edge as Partial<ConceptEdgeRecord>
  if (!candidate.id || !candidate.source || !candidate.target || !candidate.explanation) {
    throw new Error(`Edge at index ${index} is missing required fields.`)
  }
  return {
    id: String(candidate.id),
    source: String(candidate.source),
    target: String(candidate.target),
    relation: 'prerequisite_for',
    explanation: String(candidate.explanation),
    confidence: typeof candidate.confidence === 'number' ? candidate.confidence : 1,
    evidence: {
      historical_doc_id: candidate.evidence?.historical_doc_id ?? null,
      current_doc_id: candidate.evidence?.current_doc_id ?? null,
      historical_chunk_ids: asArray(candidate.evidence?.historical_chunk_ids).map(String),
      current_chunk_ids: asArray(candidate.evidence?.current_chunk_ids).map(String),
    },
    metadata: candidate.metadata ?? {},
  }
}

export function graphToFlowElements(graph: GraphData): {
  initialNodes: FlowNode[]
  initialEdges: FlowEdge[]
} {
  const initialNodes: FlowNode[] = graph.nodes.map((node) => ({
    id: node.id,
    position: { x: 0, y: 0 },
    data: {
      label: node.label,
      summary: node.summary,
      confidence: node.confidence ?? 1,
    },
  }))
  const initialEdges: FlowEdge[] = graph.edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    markerEnd: { type: MarkerType.ArrowClosed },
    animated: false,
    style: { strokeWidth: 1.75 },
    label: `${Math.round((edge.confidence ?? 1) * 100)}%`,
    data: {
      explanation: edge.explanation,
      confidence: edge.confidence ?? 1,
    },
  }))
  return { initialNodes, initialEdges }
}

export function confidenceLabel(value: number): string {
  const percentage = Math.round(Math.max(0, Math.min(1, value)) * 100)
  return `${percentage}%`
}
