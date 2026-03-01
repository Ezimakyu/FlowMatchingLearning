import type { Edge, Node } from 'reactflow'

export type GraphMetadata = Record<string, unknown>

export interface SourceMaterial {
  doc_id: string
  section_id?: string | null
  chunk_ids?: string[]
  page_numbers?: number[]
  transcript_timestamps?: string[]
  snippet?: string | null
}

export interface ConceptNodeRecord {
  id: string
  label: string
  summary: string
  aliases?: string[]
  confidence?: number
  deep_dive?: string | null
  source_material: SourceMaterial
  metadata?: GraphMetadata
}

export interface ConceptEdgeEvidence {
  historical_doc_id?: string | null
  current_doc_id?: string | null
  historical_chunk_ids?: string[]
  current_chunk_ids?: string[]
}

export interface ConceptEdgeRecord {
  id: string
  source: string
  target: string
  relation?: 'prerequisite_for'
  explanation: string
  confidence?: number
  evidence?: ConceptEdgeEvidence
  metadata?: GraphMetadata
}

export interface GraphData {
  schema_version?: string
  graph_id?: string | null
  generated_at?: string
  nodes: ConceptNodeRecord[]
  edges: ConceptEdgeRecord[]
  metadata?: GraphMetadata
}

export interface FlowNodeData {
  label: string
  summary: string
  confidence: number
}

export interface FlowEdgeData {
  explanation: string
  confidence: number
}

export type FlowNode = Node<FlowNodeData>
export type FlowEdge = Edge<FlowEdgeData>

export type LayoutMode = 'topological' | 'force'

export type SelectedDetails =
  | { kind: 'node'; node: ConceptNodeRecord }
  | { kind: 'edge'; edge: ConceptEdgeRecord }
  | null
