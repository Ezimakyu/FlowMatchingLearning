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
        <p>Select a node or edge to inspect summaries and relationships.</p>
      </aside>
    )
  }

  if (selected.kind === 'node') {
    const { node } = selected
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
    </aside>
  )
}
