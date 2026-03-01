import { confidenceLabel } from '../graphUtils'
import type { SelectedDetails } from '../types'

interface NodeDetailsPanelProps {
  selected: SelectedDetails
}

function renderStringList(values: string[] | undefined, fallback: string): string {
  if (!values || values.length === 0) {
    return fallback
  }
  return values.join(', ')
}

export function NodeDetailsPanel({ selected }: NodeDetailsPanelProps) {
  if (!selected) {
    return (
      <aside className="details-panel">
        <h3>Details</h3>
        <p>Select a node or edge to inspect evidence and explanations.</p>
      </aside>
    )
  }

  if (selected.kind === 'node') {
    const { node } = selected
    const source = node.source_material
    return (
      <aside className="details-panel">
        <h3>Node Details</h3>
        <div className="details-row">
          <span className="details-key">Label</span>
          <span className="details-value">{node.label}</span>
        </div>
        <div className="details-row">
          <span className="details-key">Confidence</span>
          <span className="details-value">{confidenceLabel(node.confidence ?? 1)}</span>
        </div>
        <div className="details-block">
          <h4>Summary</h4>
          <p>{node.summary}</p>
        </div>
        <div className="details-block">
          <h4>Aliases</h4>
          <p>{renderStringList(node.aliases, 'None')}</p>
        </div>
        <div className="details-block">
          <h4>Source Evidence</h4>
          <p>{source.snippet || 'No snippet available.'}</p>
          <p>
            <strong>Section:</strong> {source.section_id || 'Unknown'}
          </p>
          <p>
            <strong>Chunk IDs:</strong> {renderStringList(source.chunk_ids, 'None')}
          </p>
          <p>
            <strong>Pages:</strong>{' '}
            {source.page_numbers && source.page_numbers.length > 0
              ? source.page_numbers.join(', ')
              : 'None'}
          </p>
          <p>
            <strong>Timestamps:</strong>{' '}
            {renderStringList(source.transcript_timestamps, 'None')}
          </p>
        </div>
      </aside>
    )
  }

  const { edge } = selected
  return (
    <aside className="details-panel">
      <h3>Edge Details</h3>
      <div className="details-row">
        <span className="details-key">Direction</span>
        <span className="details-value">
          {edge.source} {'->'} {edge.target}
        </span>
      </div>
      <div className="details-row">
        <span className="details-key">Confidence</span>
        <span className="details-value">{confidenceLabel(edge.confidence ?? 1)}</span>
      </div>
      <div className="details-block">
        <h4>Explanation</h4>
        <p>{edge.explanation}</p>
      </div>
      <div className="details-block">
        <h4>Evidence</h4>
        <p>
          <strong>Historical Doc:</strong> {edge.evidence?.historical_doc_id || 'Unknown'}
        </p>
        <p>
          <strong>Current Doc:</strong> {edge.evidence?.current_doc_id || 'Unknown'}
        </p>
        <p>
          <strong>Historical Chunks:</strong>{' '}
          {renderStringList(edge.evidence?.historical_chunk_ids, 'None')}
        </p>
        <p>
          <strong>Current Chunks:</strong>{' '}
          {renderStringList(edge.evidence?.current_chunk_ids, 'None')}
        </p>
      </div>
    </aside>
  )
}
