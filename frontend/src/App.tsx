import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from 'react'
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
const SUPPORTED_SOURCE_EXTENSIONS = new Set(['.pdf', '.ppt', '.pptx'])
const SUPPORTED_MEDIA_EXTENSIONS = new Set([
  '.mp4',
  '.mp3',
  '.wav',
  '.m4a',
  '.mov',
  '.webm',
  '.mkv',
])

type ModelProfile = 'test' | 'demo'

interface BatchJobRow {
  jobId: string
  docId: string
  sourceFile: string
  status: string
  stage: string
  error?: string
}

function extensionOf(filename: string): string {
  const index = filename.lastIndexOf('.')
  if (index < 0) {
    return ''
  }
  return filename.slice(index).toLowerCase()
}

function stemOf(filename: string): string {
  const index = filename.lastIndexOf('.')
  if (index < 0) {
    return filename
  }
  return filename.slice(0, index)
}

function normalizePath(value: string): string {
  return value.replaceAll('\\', '/').trim().toLowerCase()
}

function relativePathOf(file: File): string {
  return file.webkitRelativePath || file.name
}

function inferDirectoryDocId(files: File[]): string {
  if (files.length === 0) {
    return 'doc_directory_batch'
  }
  const firstPath = relativePathOf(files[0])
  const firstSegment = firstPath.split('/')[0]
  return `doc_${slugify(firstSegment)}`
}

function slugify(value: string): string {
  const cleaned = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
  return cleaned || 'item'
}

function dedupeWithCounter(base: string, seen: Map<string, number>): string {
  const previous = seen.get(base) ?? 0
  seen.set(base, previous + 1)
  if (previous === 0) {
    return base
  }
  return `${base}_${previous + 1}`
}

function manifestSetFromPayload(payload: unknown): Set<string> {
  const entries = new Set<string>()
  const pushEntry = (value: unknown) => {
    if (typeof value !== 'string') {
      return
    }
    const normalized = normalizePath(value)
    if (!normalized) {
      return
    }
    entries.add(normalized)
    const segments = normalized.split('/')
    entries.add(segments[segments.length - 1])
  }

  if (Array.isArray(payload)) {
    for (const item of payload) {
      pushEntry(item)
    }
    return entries
  }

  if (!payload || typeof payload !== 'object') {
    return entries
  }
  const candidate = payload as Record<string, unknown>
  if (Array.isArray(candidate.files)) {
    for (const file of candidate.files) {
      pushEntry(file)
    }
  }
  if (Array.isArray(candidate.items)) {
    for (const item of candidate.items) {
      if (item && typeof item === 'object') {
        const sourceFile = (item as Record<string, unknown>).source_file
        pushEntry(sourceFile)
      }
    }
  }
  return entries
}

function keyPartsForFile(file: File): { parent: string; stem: string; relative: string } {
  const relative = normalizePath(relativePathOf(file))
  const segments = relative.split('/')
  const filename = segments.pop() || relative
  const parent = segments.join('/')
  return { parent, stem: stemOf(filename), relative }
}

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json()
    if (payload && typeof payload.detail === 'string') {
      return payload.detail
    }
  } catch {
    // fall back to status text below
  }
  return `Request failed with status ${response.status}.`
}

function App() {
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [allNodes, setAllNodes] = useState<FlowNode[]>([])
  const [allEdges, setAllEdges] = useState<FlowEdge[]>([])
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('topological')
  const [searchTerm, setSearchTerm] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isBatchRunning, setIsBatchRunning] = useState(false)
  const [statusText, setStatusText] = useState('Waiting to load graph_data.json...')
  const [errorText, setErrorText] = useState<string | null>(null)
  const [selected, setSelected] = useState<SelectedDetails>(null)
  const [batchModelProfile, setBatchModelProfile] = useState<ModelProfile>('test')
  const [directoryFiles, setDirectoryFiles] = useState<File[]>([])
  const [manifestSummary, setManifestSummary] = useState('No manifest filter loaded.')
  const [manifestFilter, setManifestFilter] = useState<Set<string> | null>(null)
  const [batchJobs, setBatchJobs] = useState<BatchJobRow[]>([])
  const directoryInputRef = useRef<HTMLInputElement | null>(null)
  const manifestInputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    if (directoryInputRef.current) {
      directoryInputRef.current.setAttribute('webkitdirectory', '')
      directoryInputRef.current.setAttribute('directory', '')
    }
  }, [])

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

  const loadGraphForBatchJob = useCallback(
    (jobId: string) => {
      void loadGraphFromPath(`/api/v1/jobs/${encodeURIComponent(jobId)}/graph`)
    },
    [loadGraphFromPath]
  )

  const onDirectorySelected = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.target.files ?? [])
      setDirectoryFiles(files)
      const sourceCount = files.filter((file) =>
        SUPPORTED_SOURCE_EXTENSIONS.has(extensionOf(file.name))
      ).length
      setStatusText(
        `Selected ${files.length} files from directory (${sourceCount} supported source files).`
      )
    },
    []
  )

  const onManifestSelected = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) {
      setManifestFilter(null)
      setManifestSummary('No manifest filter loaded.')
      return
    }
    try {
      const text = await file.text()
      const payload = JSON.parse(text)
      const filterSet = manifestSetFromPayload(payload)
      if (filterSet.size === 0) {
        throw new Error('Manifest did not contain any usable file names.')
      }
      setManifestFilter(filterSet)
      setManifestSummary(`Manifest loaded: ${filterSet.size} file entries.`)
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Could not parse manifest file.'
      setErrorText(message)
      setManifestFilter(null)
      setManifestSummary('Invalid manifest file.')
    }
  }, [])

  const runDirectoryBatch = useCallback(async () => {
    const sourceFiles = directoryFiles.filter((file) =>
      SUPPORTED_SOURCE_EXTENSIONS.has(extensionOf(file.name))
    )
    if (sourceFiles.length === 0) {
      setErrorText(
        `No supported source files selected. Supported: ${Array.from(SUPPORTED_SOURCE_EXTENSIONS).join(', ')}`
      )
      return
    }

    let selectedSourceFiles = sourceFiles
    if (manifestFilter && manifestFilter.size > 0) {
      selectedSourceFiles = sourceFiles.filter((file) => {
        const normalizedRelative = normalizePath(relativePathOf(file))
        const normalizedBase = normalizePath(file.name)
        return manifestFilter.has(normalizedRelative) || manifestFilter.has(normalizedBase)
      })
      if (selectedSourceFiles.length === 0) {
        setErrorText('Manifest filter removed all source files from the selected directory.')
        return
      }
    }

    const mediaByFullKey = new Map<string, File[]>()
    const mediaByStem = new Map<string, File[]>()
    for (const file of directoryFiles) {
      if (!SUPPORTED_MEDIA_EXTENSIONS.has(extensionOf(file.name))) {
        continue
      }
      const parts = keyPartsForFile(file)
      const fullKey = `${parts.parent}::${parts.stem}`
      const exactBucket = mediaByFullKey.get(fullKey) ?? []
      exactBucket.push(file)
      mediaByFullKey.set(fullKey, exactBucket)

      const stemBucket = mediaByStem.get(parts.stem) ?? []
      stemBucket.push(file)
      mediaByStem.set(parts.stem, stemBucket)
    }

    const sourceIdSeen = new Map<string, number>()
    const uploadIds: string[] = []
    const resolvedDocId = inferDirectoryDocId(selectedSourceFiles)
    setIsBatchRunning(true)
    setErrorText(null)
    setBatchJobs([])
    try {
      for (const [index, sourceFile] of selectedSourceFiles.entries()) {
        const sourceParts = keyPartsForFile(sourceFile)
        const sourceFileId = dedupeWithCounter(
          `src_${slugify(sourceParts.stem)}`,
          sourceIdSeen
        )

        const fullKey = `${sourceParts.parent}::${sourceParts.stem}`
        const exactMedia = mediaByFullKey.get(fullKey) ?? []
        const fallbackMedia = mediaByStem.get(sourceParts.stem) ?? []
        const matchedMedia =
          exactMedia.length > 0
            ? exactMedia.shift()
            : fallbackMedia.length > 0
              ? fallbackMedia.shift()
              : undefined
        if (matchedMedia) {
          const remainingExact = (mediaByFullKey.get(fullKey) ?? []).filter(
            (item) => item !== matchedMedia
          )
          if (remainingExact.length > 0) {
            mediaByFullKey.set(fullKey, remainingExact)
          } else {
            mediaByFullKey.delete(fullKey)
          }
          const remainingFallback = (mediaByStem.get(sourceParts.stem) ?? []).filter(
            (item) => item !== matchedMedia
          )
          if (remainingFallback.length > 0) {
            mediaByStem.set(sourceParts.stem, remainingFallback)
          } else {
            mediaByStem.delete(sourceParts.stem)
          }
        }

        setStatusText(
          `Uploading ${index + 1}/${selectedSourceFiles.length}: ${relativePathOf(sourceFile)}`
        )
        const uploadBody = new FormData()
        uploadBody.append('doc_id', resolvedDocId)
        uploadBody.append('source_file_id', sourceFileId)
        uploadBody.append('source_file', sourceFile)
        if (matchedMedia) {
          uploadBody.append('media_file', matchedMedia)
          uploadBody.append('media_id', `media_${sourceFileId}`)
        }

        const uploadResponse = await fetch('/api/v1/upload', {
          method: 'POST',
          body: uploadBody,
        })
        if (!uploadResponse.ok) {
          throw new Error(await parseErrorMessage(uploadResponse))
        }
        const uploadPayload = (await uploadResponse.json()) as { upload_id: string }
        uploadIds.push(uploadPayload.upload_id)
      }
      setStatusText(`Starting combined job for ${selectedSourceFiles.length} files...`)
      const startResponse = await fetch('/api/v1/jobs/start-combined', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          upload_ids: uploadIds,
          doc_id: resolvedDocId,
          model_profile: batchModelProfile,
          force_restart: true,
        }),
      })
      if (!startResponse.ok) {
        throw new Error(await parseErrorMessage(startResponse))
      }
      const startPayload = (await startResponse.json()) as {
        job: { job_id: string; status: string; stage: string; doc_id: string }
      }
      const job = startPayload.job
      setBatchJobs([
        {
          jobId: job.job_id,
          docId: job.doc_id,
          sourceFile: `${selectedSourceFiles.length} files`,
          status: job.status,
          stage: job.stage,
        },
      ])
      setStatusText(`Started combined job ${job.job_id} for ${selectedSourceFiles.length} files.`)
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to start directory batch.'
      setErrorText(message)
      setStatusText('Directory batch failed.')
    } finally {
      setIsBatchRunning(false)
    }
  }, [batchModelProfile, directoryFiles, manifestFilter])

  const refreshBatchStatuses = useCallback(async () => {
    if (batchJobs.length === 0) {
      return
    }
    setIsLoading(true)
    setErrorText(null)
    try {
      const updated = await Promise.all(
        batchJobs.map(async (job) => {
          const response = await fetch(`/api/v1/jobs/${encodeURIComponent(job.jobId)}`)
          if (!response.ok) {
            return {
              ...job,
              error: await parseErrorMessage(response),
            }
          }
          const payload = (await response.json()) as {
            job: { status: string; stage: string; doc_id: string; error?: string | null }
          }
          return {
            ...job,
            docId: payload.job.doc_id,
            status: payload.job.status,
            stage: payload.job.stage,
            error: payload.job.error || undefined,
          }
        })
      )
      setBatchJobs(updated)
      setStatusText(`Refreshed ${updated.length} job statuses.`)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Could not refresh job status.'
      setErrorText(message)
    } finally {
      setIsLoading(false)
    }
  }, [batchJobs])

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
          <h1>FlowMatchingLearning</h1>
          <p>React Flow visualization for `graph_data.json`.</p>
        </div>
        <div className="toolbar-controls">
          <div className="toolbar-group toolbar-group-left">
            <button
              type="button"
              onClick={() => directoryInputRef.current?.click()}
              disabled={isBatchRunning}
            >
              Select directory
            </button>
            <input
              ref={directoryInputRef}
              type="file"
              multiple
              onChange={onDirectorySelected}
              className="visually-hidden"
            />
            <button
              type="button"
              onClick={() => manifestInputRef.current?.click()}
              disabled={isBatchRunning}
            >
              Load file-list JSON (optional)
            </button>
            <input
              ref={manifestInputRef}
              type="file"
              accept=".json,application/json"
              onChange={onManifestSelected}
              className="visually-hidden"
            />
            <button
              type="button"
              className="primary-button"
              onClick={runDirectoryBatch}
              disabled={isBatchRunning || directoryFiles.length === 0}
            >
              Run single combined job
            </button>
          </div>
          <div className="toolbar-group toolbar-group-right">
            <button type="button" onClick={reloadLocalGraph} disabled={isLoading}>
              Reload local graph_data.json
            </button>
            <button
              type="button"
              onClick={refreshBatchStatuses}
              disabled={isBatchRunning || batchJobs.length === 0}
            >
              Refresh job statuses
            </button>
            <label>
              Batch model
              <select
                value={batchModelProfile}
                onChange={(event) => setBatchModelProfile(event.target.value as ModelProfile)}
              >
                <option value="test">test</option>
                <option value="demo">demo</option>
              </select>
            </label>
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
        </div>
        <div className="toolbar-status">
          <span>{statusText}</span>
          <span>
            Directory selection: {directoryFiles.length} files ({manifestSummary})
          </span>
          <span>
            Showing {visibleNodes.length}/{allNodes.length} nodes, {visibleEdges.length}/
            {allEdges.length} edges
          </span>
          {errorText ? <span className="error-text">{errorText}</span> : null}
        </div>
      </header>
      {batchJobs.length > 0 ? (
        <section className="batch-jobs">
          <h2>Batch Jobs</h2>
          <div className="batch-jobs-list">
            {batchJobs.map((job) => (
              <div key={job.jobId} className="batch-job-row">
                <div className="batch-job-main">
                  <span className="batch-job-source">{job.sourceFile}</span>
                  <span className="batch-job-id">{job.jobId}</span>
                </div>
                <div className="batch-job-status">
                  <span className={`status-chip status-${job.status.toLowerCase()}`}>
                    {job.status}
                  </span>
                  <span className="stage-chip">{job.stage}</span>
                  <button type="button" onClick={() => loadGraphForBatchJob(job.jobId)}>
                    Load graph
                  </button>
                </div>
                {job.error ? <div className="batch-job-error">{job.error}</div> : null}
              </div>
            ))}
          </div>
        </section>
      ) : null}
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
