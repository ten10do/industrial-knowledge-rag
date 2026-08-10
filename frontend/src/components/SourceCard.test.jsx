import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import SourceCard from './SourceCard'

describe('SourceCard', () => {
  it('renders the backend source contract including score', () => {
    render(
      <SourceCard
        index={0}
        source={{
          citation_id: 'S1',
          source: 'course.pdf',
          page: 3,
          score: 9.25,
          content: 'PLC 扫描周期参考内容',
          equipment_model: 'IK-PLC-100',
          section: '1.1 通信参数',
          knowledge_type: 'parameter',
        }}
      />,
    )

    expect(screen.getByText('course.pdf')).toBeInTheDocument()
    expect(screen.getByText('[S1]')).toBeInTheDocument()
    expect(screen.getByText('第 3 页')).toBeInTheDocument()
    expect(screen.getByText('距离 9.2500')).toBeInTheDocument()
    expect(screen.getByText('PLC 扫描周期参考内容')).toBeInTheDocument()
    expect(screen.getByText('IK-PLC-100')).toBeInTheDocument()
    expect(screen.getByText('1.1 通信参数')).toBeInTheDocument()
    expect(screen.getByText('parameter')).toBeInTheDocument()
  })
})
