import { useCallback, useEffect, useMemo, useState } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  type EdgeMouseHandler,
  type NodeMouseHandler,
} from 'reactflow'
import 'reactflow/dist/style.css'

import './App.css'
import { NodeDetailsPanel } from './components/NodeDetailsPanel'
import { graphToFlowElements, parseGraphData } from './graphUtils'
import { applyLayoutMode } from './layout'
import type {
  ConceptEdgeRecord,
  ConceptNodeRecord,
  FlowEdge,
  FlowNode,
  GraphData,
  LayoutMode,
  SelectedDetails,
} from './types'

const DEFAULT_GRAPH_PATH = '/graph_data.json'

function App() {
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [allNodes, setAllNodes] = useState<FlowNode[]>([])
  const [allEdges, setAllEdges] = useState<FlowEdge[]>([])
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('topological')
  const [searchTerm, setSearchTerm] = useState('')
  const [jobIdInput, setJobIdInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [statusText, setStatusText] = useState('Waiting to load graph_data.json...')
  const [errorText, setErrorText] = useState<string | null>(null)
  const [selected, setSelected] = useState<SelectedDetails>(null)

  const nodeById = useMemo(() => {
    const map = new Map<string, ConceptNodeRecord>()
    for (const node of graphData?.nodes ?? []) {
      map.set(node.id, node)
    }
    return map
  }, [graphData])

  const edgeById = useMemo(() => {
    const map = new Map<string, ConceptEdgeRecord>()
    for (const edge of graphData?.edges ?? []) {
      map.set(edge.id, edge)
    }
    return map
  }, [graphData])

  const applyGraphData = useCallback(
    (graph: GraphData) => {
      const { initialNodes, initialEdges } = graphToFlowElements(graph)
      const laidOutNodes = applyLayoutMode({
        mode: layoutMode,
        nodes: initialNodes,
        edges: initialEdges,
      })
      setAllNodes(laidOutNodes)
      setAllEdges(initialEdges)
      setStatusText(
        `Loaded ${graph.nodes.length} nodes and ${graph.edges.length} edges (${layoutMode} layout).`
      )
    },
    [layoutMode]
  )

  const loadGraphFromPath = useCallback(
    async (path: string) => {
      setIsLoading(true)
      setErrorText(null)
      try {
        const response = await fetch(path)
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}.`)
        }
        const payload = await response.json()
        const parsed = parseGraphData(payload)
        setGraphData(parsed)
        setSelected(null)
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : 'Unexpected error while loading graph JSON.'
        setErrorText(message)
        setStatusText('Failed to load graph data.')
      } finally {
        setIsLoading(false)
      }
    },
    []
  )

  const reloadLocalGraph = useCallback(() => {
    void loadGraphFromPath(DEFAULT_GRAPH_PATH)
  }, [loadGraphFromPath])

  const loadGraphFromJob = useCallback(() => {
    const jobId = jobIdInput.trim()
    if (!jobId) {
      setErrorText('Enter a job id before loading from API.')
      return
    }
    void loadGraphFromPath(`/api/v1/jobs/${encodeURIComponent(jobId)}/graph`)
  }, [jobIdInput, loadGraphFromPath])

  useEffect(() => {
    reloadLocalGraph()
  }, [reloadLocalGraph])

  useEffect(() => {
    if (!graphData) {
      return
    }
    applyGraphData(graphData)
  }, [graphData, applyGraphData])

  const matchedNodeIds = useMemo(() => {
    if (!graphData) {
      return new Set<string>()
    }
    const query = searchTerm.trim().toLowerCase()
    if (!query) {
      return new Set(graphData.nodes.map((node) => node.id))
    }
    const matching = new Set<string>()
    for (const node of graphData.nodes) {
      const haystack = [
        node.label,
        node.summary,
        ...(node.aliases ?? []),
        node.source_material.section_id ?? '',
        node.source_material.snippet ?? '',
      ]
        .join(' ')
        .toLowerCase()
      if (haystack.includes(query)) {
        matching.add(node.id)
      }
    }
    return matching
  }, [graphData, searchTerm])

  const visibleNodes = useMemo(
    () =>
      allNodes
        .filter((node) => matchedNodeIds.has(node.id))
        .map((node) => ({
          ...node,
          selected: selected?.kind === 'node' && selected.node.id === node.id,
        })),
    [allNodes, matchedNodeIds, selected]
  )

  const visibleEdges = useMemo(
    () =>
      allEdges
        .filter(
          (edge) => matchedNodeIds.has(edge.source) && matchedNodeIds.has(edge.target)
        )
        .map((edge) => ({
          ...edge,
          selected: selected?.kind === 'edge' && selected.edge.id === edge.id,
        })),
    [allEdges, matchedNodeIds, selected]
  )

  const onNodeClick = useCallback<NodeMouseHandler>(
    (_event, node) => {
      const detail = nodeById.get(node.id)
      if (!detail) {
        return
      }
      setSelected({ kind: 'node', node: detail })
    },
    [nodeById]
  )

  const onEdgeClick = useCallback<EdgeMouseHandler>(
    (_event, edge) => {
      const detail = edgeById.get(edge.id)
      if (!detail) {
        return
      }
      setSelected({ kind: 'edge', edge: detail })
    },
    [edgeById]
  )

  return (
    <div className="app-shell">
      <header className="toolbar">
        <div className="toolbar-title">
          <h1>Prerequisite Graph Viewer</h1>
          <p>React Flow visualization for `graph_data.json`.</p>
        </div>
        <div className="toolbar-controls">
          <button type="button" onClick={reloadLocalGraph} disabled={isLoading}>
            Reload local graph_data.json
          </button>
          <div className="job-loader">
            <input
              type="text"
              value={jobIdInput}
              onChange={(event) => setJobIdInput(event.target.value)}
              placeholder="Job ID (optional API load)"
            />
            <button type="button" onClick={loadGraphFromJob} disabled={isLoading}>
              Load from API
            </button>
          </div>
          <label>
            Layout
            <select
              value={layoutMode}
              onChange={(event) => setLayoutMode(event.target.value as LayoutMode)}
            >
              <option value="topological">Topological</option>
              <option value="force">Force</option>
            </select>
          </label>
          <label>
            Search / Filter
            <input
              type="text"
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              placeholder="Find concepts"
            />
          </label>
        </div>
        <div className="toolbar-status">
          <span>{statusText}</span>
          <span>
            Showing {visibleNodes.length}/{allNodes.length} nodes, {visibleEdges.length}/
            {allEdges.length} edges
          </span>
          {errorText ? <span className="error-text">{errorText}</span> : null}
        </div>
      </header>
      <main className="content">
        <div className="flow-wrapper">
          <ReactFlow
            nodes={visibleNodes}
            edges={visibleEdges}
            onNodeClick={onNodeClick}
            onEdgeClick={onEdgeClick}
            onPaneClick={() => setSelected(null)}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            minZoom={0.15}
            maxZoom={2}
            nodesDraggable={false}
            proOptions={{ hideAttribution: true }}
          >
            <MiniMap />
            <Controls />
            <Background gap={18} />
          </ReactFlow>
        </div>
        <NodeDetailsPanel selected={selected} />
      </main>
    </div>
  )
}

export default App
