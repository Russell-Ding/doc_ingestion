import React, { useState, useEffect } from 'react';
import {
  Form,
  Input,
  Button,
  Space,
  Card,
  Select,
  Switch,
  Slider,
  Typography,
  Alert,
  Row,
  Col,
  Tag,
  Tooltip,
  Divider
} from 'antd';
import {
  SaveOutlined,
  CloseOutlined,
  InfoCircleOutlined,
  BulbOutlined,
  FileTextOutlined,
  TableOutlined,
  CalculatorOutlined
} from '@ant-design/icons';

import { ReportSegment } from '../../types/report';
import { useDocuments } from '../../hooks/useDocuments';

const { TextArea } = Input;
const { Option } = Select;
const { Title, Text, Paragraph } = Typography;

interface SegmentEditorProps {
  segment: ReportSegment;
  onSave: (segment: ReportSegment) => void;
  onCancel: () => void;
}

const PROMPT_TEMPLATES = {
  financial_summary: {
    name: 'Financial Summary',
    icon: <CalculatorOutlined />,
    prompt: 'Please provide a comprehensive financial summary based on the available financial documents. Include key metrics such as revenue, profitability, cash flow, and financial ratios. Highlight any significant trends or concerns.',
    description: 'Generates a summary of financial performance and key metrics'
  },
  risk_assessment: {
    name: 'Risk Assessment',
    icon: <FileTextOutlined />,
    prompt: 'Analyze the credit risk factors based on the provided documentation. Evaluate the borrower\'s creditworthiness, including financial stability, industry risks, market position, and management quality. Provide a risk rating and key concerns.',
    description: 'Evaluates credit risk factors and provides risk rating'
  },
  cash_flow_analysis: {
    name: 'Cash Flow Analysis',
    icon: <TableOutlined />,
    prompt: 'Analyze the cash flow patterns from the financial statements and projections. Evaluate the quality and sustainability of cash flows, seasonal variations, and cash flow coverage ratios.',
    description: 'Detailed analysis of cash flow patterns and sustainability'
  },
  industry_analysis: {
    name: 'Industry Analysis',
    icon: <FileTextOutlined />,
    prompt: 'Provide an analysis of the industry in which the borrower operates. Include market conditions, competitive landscape, regulatory environment, and industry-specific risks and opportunities.',
    description: 'Analysis of industry conditions and market environment'
  },
  management_assessment: {
    name: 'Management Assessment',
    icon: <FileTextOutlined />,
    prompt: 'Evaluate the management team\'s experience, track record, and ability to execute business plans. Consider leadership changes, key person risks, and management depth.',
    description: 'Assessment of management team quality and experience'
  }
};

const DOCUMENT_TYPES = [
  { value: 'financial_statements', label: 'Financial Statements', icon: <TableOutlined /> },
  { value: 'cash_flow', label: 'Cash Flow Statements', icon: <CalculatorOutlined /> },
  { value: 'contracts', label: 'Contracts & Agreements', icon: <FileTextOutlined /> },
  { value: 'business_plan', label: 'Business Plan', icon: <BulbOutlined /> },
  { value: 'tax_returns', label: 'Tax Returns', icon: <FileTextOutlined /> },
  { value: 'bank_statements', label: 'Bank Statements', icon: <TableOutlined /> },
  { value: 'projections', label: 'Financial Projections', icon: <CalculatorOutlined /> },
  { value: 'collateral', label: 'Collateral Documentation', icon: <FileTextOutlined /> }
];

export const SegmentEditor: React.FC<SegmentEditorProps> = ({
  segment,
  onSave,
  onCancel
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  const { documents } = useDocuments();

  useEffect(() => {
    form.setFieldsValue({
      name: segment.name || '',
      description: segment.description || '',
      prompt: segment.prompt || '',
      required_document_types: segment.required_document_types || [],
      generation_settings: {
        max_tokens: segment.generation_settings?.max_tokens || 2000,
        temperature: segment.generation_settings?.temperature || 0.7,
        include_tables: segment.generation_settings?.include_tables !== false,
        validation_enabled: segment.generation_settings?.validation_enabled !== false,
        ...segment.generation_settings
      }
    });
  }, [form, segment]);

  const handleTemplateSelect = (templateKey: string) => {
    const template = PROMPT_TEMPLATES[templateKey as keyof typeof PROMPT_TEMPLATES];
    if (template) {
      form.setFieldsValue({
        name: form.getFieldValue('name') || template.name,
        prompt: template.prompt,
        description: template.description
      });
      setSelectedTemplate(templateKey);
    }
  };

  const handleSave = async () => {
    try {
      setLoading(true);
      const values = await form.validateFields();
      
      const updatedSegment: ReportSegment = {
        ...segment,
        ...values,
        updated_date: new Date().toISOString()
      };

      onSave(updatedSegment);
    } catch (error) {
      console.error('Form validation failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const availableDocuments = documents.filter(doc => 
    doc.processing_status === 'completed'
  );

  return (
    <div className="segment-editor">
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSave}
        scrollToFirstError
      >
        <Row gutter={[24, 16]}>
          <Col span={24}>
            <Card size="small" title="Basic Information">
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="name"
                    label="Segment Name"
                    rules={[{ required: true, message: 'Segment name is required' }]}
                  >
                    <Input 
                      placeholder="e.g., Financial Summary, Risk Assessment"
                      prefix={<FileTextOutlined />}
                    />
                  </Form.Item>
                </Col>
                
                <Col span={12}>
                  <Form.Item
                    name="description"
                    label="Description"
                  >
                    <Input 
                      placeholder="Brief description of what this segment covers"
                    />
                  </Form.Item>
                </Col>
              </Row>
            </Card>
          </Col>

          <Col span={24}>
            <Card 
              size="small" 
              title="Prompt Templates"
              extra={
                <Tooltip title="Use pre-built templates for common report sections">
                  <InfoCircleOutlined />
                </Tooltip>
              }
            >
              <Text type="secondary" style={{ marginBottom: 16, display: 'block' }}>
                Choose a template to get started, or write a custom prompt below:
              </Text>
              
              <Row gutter={[8, 8]}>
                {Object.entries(PROMPT_TEMPLATES).map(([key, template]) => (
                  <Col key={key}>
                    <Button
                      size="small"
                      type={selectedTemplate === key ? 'primary' : 'default'}
                      onClick={() => handleTemplateSelect(key)}
                      style={{ height: 'auto', padding: '8px 12px' }}
                    >
                      <Space direction="vertical" size="small" align="center">
                        {template.icon}
                        <Text style={{ fontSize: '12px' }}>{template.name}</Text>
                      </Space>
                    </Button>
                  </Col>
                ))}
              </Row>
            </Card>
          </Col>

          <Col span={24}>
            <Card size="small" title="Content Generation Prompt">
              <Form.Item
                name="prompt"
                rules={[
                  { required: true, message: 'Prompt is required' },
                  { min: 50, message: 'Prompt should be at least 50 characters' }
                ]}
              >
                <TextArea
                  rows={8}
                  placeholder="Describe what you want this segment to include. Be specific about the analysis, data, and insights you need. The AI will use your uploaded documents to generate relevant content based on this prompt."
                  showCount
                  maxLength={2000}
                />
              </Form.Item>
              
              <Alert
                type="info"
                showIcon
                icon={<BulbOutlined />}
                message="Prompt Writing Tips"
                description={
                  <ul style={{ marginBottom: 0, paddingLeft: 20 }}>
                    <li>Be specific about what analysis you want (financial ratios, trends, comparisons)</li>
                    <li>Mention specific data points or metrics you need included</li>
                    <li>Specify the tone (executive summary, detailed analysis, risk-focused)</li>
                    <li>Include formatting preferences (tables, bullet points, paragraphs)</li>
                  </ul>
                }
              />
            </Card>
          </Col>

          <Col span={24}>
            <Card size="small" title="Document Requirements">
              <Form.Item
                name="required_document_types"
                label="Required Document Types"
              >
                <Select
                  mode="multiple"
                  placeholder="Select document types needed for this segment"
                  allowClear
                  showSearch={false}
                >
                  {DOCUMENT_TYPES.map(type => (
                    <Option key={type.value} value={type.value}>
                      <Space>
                        {type.icon}
                        {type.label}
                      </Space>
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              {availableDocuments.length > 0 && (
                <div>
                  <Text strong>Available Documents:</Text>
                  <div style={{ marginTop: 8 }}>
                    {availableDocuments.map(doc => (
                      <Tag key={doc.id} style={{ margin: '2px 4px 2px 0' }}>
                        {doc.name}
                      </Tag>
                    ))}
                  </div>
                </div>
              )}
            </Card>
          </Col>

          <Col span={24}>
            <Card 
              size="small" 
              title="Generation Settings"
              extra={
                <Button 
                  type="link" 
                  size="small"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                >
                  {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
                </Button>
              }
            >
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name={['generation_settings', 'include_tables']}
                    label="Include Tables"
                    valuePropName="checked"
                  >
                    <Switch 
                      checkedChildren="Yes" 
                      unCheckedChildren="No"
                    />
                  </Form.Item>
                </Col>
                
                <Col span={12}>
                  <Form.Item
                    name={['generation_settings', 'validation_enabled']}
                    label="Content Validation"
                    valuePropName="checked"
                  >
                    <Switch 
                      checkedChildren="Enabled" 
                      unCheckedChildren="Disabled"
                    />
                  </Form.Item>
                </Col>
              </Row>

              {showAdvanced && (
                <>
                  <Divider />
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        name={['generation_settings', 'max_tokens']}
                        label="Max Length (tokens)"
                      >
                        <Slider
                          min={500}
                          max={4000}
                          step={100}
                          marks={{
                            500: '500',
                            2000: '2K',
                            4000: '4K'
                          }}
                          tooltip={{ formatter: (value) => `${value} tokens (~${Math.round((value || 0) * 0.75)} words)` }}
                        />
                      </Form.Item>
                    </Col>
                    
                    <Col span={12}>
                      <Form.Item
                        name={['generation_settings', 'temperature']}
                        label="Creativity Level"
                      >
                        <Slider
                          min={0.1}
                          max={1.0}
                          step={0.1}
                          marks={{
                            0.1: 'Precise',
                            0.5: 'Balanced',
                            1.0: 'Creative'
                          }}
                          tooltip={{ formatter: (value) => `Temperature: ${value}` }}
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                </>
              )}
            </Card>
          </Col>
        </Row>

        <Divider />

        <Row justify="end">
          <Col>
            <Space>
              <Button onClick={onCancel}>
                <CloseOutlined />
                Cancel
              </Button>
              
              <Button 
                type="primary" 
                htmlType="submit"
                loading={loading}
              >
                <SaveOutlined />
                Save Segment
              </Button>
            </Space>
          </Col>
        </Row>
      </Form>
    </div>
  );
};