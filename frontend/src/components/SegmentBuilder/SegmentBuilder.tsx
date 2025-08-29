import React, { useState, useCallback } from 'react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import {
  Layout,
  Typography,
  Button,
  Space,
  Card,
  Modal,
  message,
  Divider,
  Row,
  Col,
  Tooltip,
  Tag
} from 'antd';
import {
  PlusOutlined,
  SaveOutlined,
  PlayCircleOutlined,
  EyeOutlined,
  SettingOutlined
} from '@ant-design/icons';

import { SegmentList } from './SegmentList';
import { SegmentEditor } from './SegmentEditor';
import { ReportPreview } from './ReportPreview';
import { DocumentUpload } from '../DocumentUpload/DocumentUpload';
import { GenerationProgress } from './GenerationProgress';
import { useSegments } from '../../hooks/useSegments';
import { useReports } from '../../hooks/useReports';
import { ReportSegment, ReportGenerationStatus } from '../../types/report';

const { Header, Content, Sider } = Layout;
const { Title, Text } = Typography;

interface SegmentBuilderProps {
  reportId?: string;
}

export const SegmentBuilder: React.FC<SegmentBuilderProps> = ({ reportId }) => {
  const [selectedSegment, setSelectedSegment] = useState<ReportSegment | null>(null);
  const [isEditorVisible, setIsEditorVisible] = useState(false);
  const [isPreviewVisible, setIsPreviewVisible] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationStatus, setGenerationStatus] = useState<ReportGenerationStatus | null>(null);

  const {
    segments,
    loading: segmentsLoading,
    addSegment,
    updateSegment,
    deleteSegment,
    reorderSegments
  } = useSegments(reportId);

  const {
    generateReport,
    validateReport,
    exportReport
  } = useReports();

  const handleAddSegment = useCallback(() => {
    const newSegment: Partial<ReportSegment> = {
      name: `New Segment ${segments.length + 1}`,
      description: '',
      prompt: '',
      order_index: segments.length,
      required_document_types: [],
      content_status: 'pending'
    };

    setSelectedSegment(newSegment as ReportSegment);
    setIsEditorVisible(true);
  }, [segments.length]);

  const handleEditSegment = useCallback((segment: ReportSegment) => {
    setSelectedSegment(segment);
    setIsEditorVisible(true);
  }, []);

  const handleSaveSegment = useCallback(async (segment: ReportSegment) => {
    try {
      if (segment.id) {
        await updateSegment(segment.id, segment);
        message.success('Segment updated successfully');
      } else {
        await addSegment(segment);
        message.success('Segment created successfully');
      }
      
      setIsEditorVisible(false);
      setSelectedSegment(null);
    } catch (error) {
      message.error('Failed to save segment');
      console.error('Error saving segment:', error);
    }
  }, [addSegment, updateSegment]);

  const handleDeleteSegment = useCallback(async (segmentId: string) => {
    try {
      await deleteSegment(segmentId);
      message.success('Segment deleted successfully');
    } catch (error) {
      message.error('Failed to delete segment');
      console.error('Error deleting segment:', error);
    }
  }, [deleteSegment]);

  const handleGenerateReport = useCallback(async () => {
    if (segments.length === 0) {
      message.warning('Please add at least one segment before generating the report');
      return;
    }

    const incompleteSegments = segments.filter(s => !s.prompt.trim());
    if (incompleteSegments.length > 0) {
      message.warning('Please complete all segment prompts before generating the report');
      return;
    }

    setIsGenerating(true);
    setGenerationStatus({
      status: 'generating',
      progress: 0,
      current_segment: 0,
      total_segments: segments.length,
      message: 'Starting report generation...'
    });

    try {
      if (!reportId) {
        throw new Error('Report ID is required');
      }

      // Start the generation process
      await generateReport(reportId, {
        segments: segments.map(s => s.id),
        validation_enabled: true,
        export_format: 'word'
      });

      // Poll for status updates (in real implementation, use WebSocket)
      const pollInterval = setInterval(async () => {
        // This would be replaced with actual status polling
        setGenerationStatus(prev => {
          if (!prev) return null;
          
          const newProgress = Math.min(prev.progress + 10, 100);
          const newCurrentSegment = Math.floor((newProgress / 100) * segments.length);
          
          if (newProgress >= 100) {
            clearInterval(pollInterval);
            setIsGenerating(false);
            message.success('Report generated successfully!');
            return {
              status: 'completed',
              progress: 100,
              current_segment: segments.length,
              total_segments: segments.length,
              message: 'Report generation completed'
            };
          }
          
          return {
            ...prev,
            progress: newProgress,
            current_segment: newCurrentSegment,
            message: `Generating segment ${newCurrentSegment + 1}: ${segments[newCurrentSegment]?.name || 'Unknown'}`
          };
        });
      }, 2000);

    } catch (error) {
      setIsGenerating(false);
      setGenerationStatus(null);
      message.error('Failed to generate report');
      console.error('Error generating report:', error);
    }
  }, [segments, reportId, generateReport]);

  const handleExportReport = useCallback(async (format: 'word' | 'pdf') => {
    try {
      if (!reportId) {
        throw new Error('Report ID is required');
      }

      const result = await exportReport(reportId, format);
      
      // Create download link
      const link = document.createElement('a');
      link.href = result.download_url;
      link.download = result.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      message.success(`Report exported as ${format.toUpperCase()}`);
    } catch (error) {
      message.error(`Failed to export report as ${format.toUpperCase()}`);
      console.error('Export error:', error);
    }
  }, [reportId, exportReport]);

  const completedSegments = segments.filter(s => s.content_status === 'completed').length;
  const totalSegments = segments.length;

  return (
    <DndProvider backend={HTML5Backend}>
      <Layout className="segment-builder" style={{ height: '100vh' }}>
        <Header style={{ background: '#fff', padding: '0 24px', boxShadow: '0 1px 4px rgba(0,0,0,0.1)' }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Title level={2} style={{ margin: 0 }}>
                Report Builder
              </Title>
              <Text type="secondary">
                {totalSegments > 0 && (
                  <>
                    {completedSegments}/{totalSegments} segments completed
                    {completedSegments > 0 && (
                      <Tag color="green" style={{ marginLeft: 8 }}>
                        {Math.round((completedSegments / totalSegments) * 100)}% Complete
                      </Tag>
                    )}
                  </>
                )}
              </Text>
            </Col>
            
            <Col>
              <Space>
                <Tooltip title="Preview Report">
                  <Button
                    icon={<EyeOutlined />}
                    onClick={() => setIsPreviewVisible(true)}
                    disabled={segments.length === 0}
                  >
                    Preview
                  </Button>
                </Tooltip>
                
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={handleGenerateReport}
                  loading={isGenerating}
                  disabled={segments.length === 0}
                >
                  Generate Report
                </Button>
                
                <Button.Group>
                  <Button
                    onClick={() => handleExportReport('word')}
                    disabled={completedSegments === 0}
                  >
                    Export Word
                  </Button>
                  <Button
                    onClick={() => handleExportReport('pdf')}
                    disabled={completedSegments === 0}
                  >
                    Export PDF
                  </Button>
                </Button.Group>
              </Space>
            </Col>
          </Row>
        </Header>

        <Layout>
          <Sider width={300} style={{ background: '#fff', borderRight: '1px solid #f0f0f0' }}>
            <div style={{ padding: '16px' }}>
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                <Card size="small" title="Documents">
                  <DocumentUpload reportId={reportId} />
                </Card>
                
                <Card size="small" title="Report Segments">
                  <Button
                    type="dashed"
                    icon={<PlusOutlined />}
                    block
                    onClick={handleAddSegment}
                    style={{ marginBottom: 16 }}
                  >
                    Add Segment
                  </Button>
                  
                  <SegmentList
                    segments={segments}
                    loading={segmentsLoading}
                    onEdit={handleEditSegment}
                    onDelete={handleDeleteSegment}
                    onReorder={reorderSegments}
                  />
                </Card>
              </Space>
            </div>
          </Sider>

          <Content style={{ padding: '24px', background: '#f5f5f5' }}>
            {isGenerating && generationStatus ? (
              <GenerationProgress status={generationStatus} />
            ) : (
              <Card style={{ height: '100%' }}>
                <div style={{ 
                  height: '100%', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  flexDirection: 'column'
                }}>
                  <Title level={3} type="secondary">
                    Welcome to the Report Builder
                  </Title>
                  <Text type="secondary" style={{ fontSize: '16px', textAlign: 'center', maxWidth: '500px' }}>
                    Start by uploading your documents, then add report segments to define what content you want to generate.
                    Each segment can have custom prompts and will automatically find relevant information from your documents.
                  </Text>
                  
                  <Space style={{ marginTop: '32px' }}>
                    <Button type="primary" size="large" onClick={handleAddSegment}>
                      <PlusOutlined />
                      Add Your First Segment
                    </Button>
                  </Space>
                </div>
              </Card>
            )}
          </Content>
        </Layout>

        {/* Segment Editor Modal */}
        <Modal
          title={selectedSegment?.id ? 'Edit Segment' : 'Create New Segment'}
          open={isEditorVisible}
          onCancel={() => {
            setIsEditorVisible(false);
            setSelectedSegment(null);
          }}
          footer={null}
          width={800}
          destroyOnClose
        >
          {selectedSegment && (
            <SegmentEditor
              segment={selectedSegment}
              onSave={handleSaveSegment}
              onCancel={() => {
                setIsEditorVisible(false);
                setSelectedSegment(null);
              }}
            />
          )}
        </Modal>

        {/* Report Preview Modal */}
        <Modal
          title="Report Preview"
          open={isPreviewVisible}
          onCancel={() => setIsPreviewVisible(false)}
          footer={null}
          width="90%"
          style={{ top: 20 }}
        >
          <ReportPreview segments={segments} />
        </Modal>
      </Layout>
    </DndProvider>
  );
};