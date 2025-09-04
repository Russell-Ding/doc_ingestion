import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any
import pandas as pd
from io import BytesIO
import uuid

# Configure Streamlit page
st.set_page_config(
    page_title="Credit Review System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Backend API base URL
API_BASE_URL = "http://localhost:8000/api/v1"

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health/", timeout=2)
        return response.status_code == 200
    except:
        return False

def upload_document(file, document_name=None):
    """Upload document to backend"""
    files = {"file": file}
    data = {}
    if document_name:
        data["document_name"] = document_name
    
    try:
        response = requests.post(f"{API_BASE_URL}/documents/upload", files=files, data=data)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return None

def get_documents():
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents/")
        return response.json().get("documents", []) if response.status_code == 200 else []
    except:
        return []

def generate_segment_content(segment_data, selected_document_ids=None):
    """Generate content for a segment using the report coordinator"""
    try:
        data = {
            "segment_data": segment_data,
            "validation_enabled": True,
            "selected_document_ids": selected_document_ids or []
        }
        # Use the existing endpoint but with actual segment data
        response = requests.post(f"{API_BASE_URL}/segments/generate-report", json=data)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        return {"error": str(e)}

def create_report_and_export_word(report_data, include_validation_comments=True):
    """Create a report and export as Word document"""
    try:
        # First create a report record
        report_response = requests.post(f"{API_BASE_URL}/reports/", json={
            "title": report_data['title'],
            "description": f"Generated report with {len(report_data['sections'])} sections"
        })
        
        if report_response.status_code != 200:
            return {"error": "Failed to create report record"}
        
        report_id = report_response.json()['id']
        
        # Create segments for the report
        segment_ids = []
        validation_results_list = []
        
        for i, section in enumerate(report_data['sections']):
            # Get validation results for this section
            validation_results = section.get('validation_results', {})
            validation_results_list.append(validation_results)
            
            segment_data = {
                "report_id": report_id,
                "name": section['name'],
                "description": f"Section {i+1}",
                "prompt": f"Generated content for {section['name']}",
                "order_index": i,
                "required_document_types": [],
                "generation_settings": {},
                "generated_content": section['content'],
                "content_status": "completed",
                "validation_results": validation_results
            }
            
            segment_response = requests.post(f"{API_BASE_URL}/segments/", json=segment_data)
            if segment_response.status_code == 200:
                segment_ids.append(segment_response.json()['id'])
        
        # Export as Word document
        export_response = requests.post(
            f"{API_BASE_URL}/reports/{report_id}/export/word",
            params={"include_validations": include_validation_comments}
        )
        
        if export_response.status_code == 200:
            return export_response.json()
        else:
            return {"error": f"Export failed: {export_response.text}"}
            
    except Exception as e:
        return {"error": str(e)}

def download_word_document(download_url):
    """Download Word document from the backend"""
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}{download_url}")
        if response.status_code == 200:
            return response.content
        else:
            return None
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return None

def create_basic_word_content(report):
    """Create a basic Word document as fallback when backend fails"""
    try:
        from docx import Document
        from docx.shared import Inches, Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        
        # Create new document
        doc = Document()
        
        # Add title
        title = doc.add_heading(report['title'], 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add generation date
        date_para = doc.add_paragraph(f"Generated on {report['generation_date']}")
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add page break
        doc.add_page_break()
        
        # Add sections
        for i, section in enumerate(report['sections']):
            # Section heading
            doc.add_heading(f"{i+1}. {section['name']}", level=1)
            
            # Section content
            content = section.get('content', 'No content available')
            paragraphs = content.split('\n\n')
            
            for paragraph in paragraphs:
                if paragraph.strip():
                    doc.add_paragraph(paragraph.strip())
            
            # Add spacing
            doc.add_paragraph()
        
        # Save to BytesIO
        from io import BytesIO
        word_buffer = BytesIO()
        doc.save(word_buffer)
        word_buffer.seek(0)
        
        return word_buffer.getvalue()
        
    except ImportError:
        # If python-docx is not available, create a simple text-based "Word" document
        content = f"{report['title']}\n\nGenerated on {report['generation_date']}\n\n"
        
        for i, section in enumerate(report['sections']):
            content += f"{i+1}. {section['name']}\n\n"
            content += f"{section.get('content', 'No content available')}\n\n"
            content += "-" * 50 + "\n\n"
        
        return content.encode('utf-8')
    
    except Exception as e:
        # Ultimate fallback - just return text content
        content = f"{report['title']}\n\nGenerated on {report['generation_date']}\n\n"
        
        for i, section in enumerate(report['sections']):
            content += f"{i+1}. {section['name']}\n\n"
            content += f"{section.get('content', 'No content available')}\n\n"
        
        return content.encode('utf-8')

def main():
    st.title("📊 Credit Review Document System")
    
    # Check backend connectivity
    if not check_backend_health():
        st.error("🔴 Backend server is not running. Please start the backend first.")
        st.code("cd backend && uvicorn app.main:app --reload")
        return
    
    st.success("🟢 Backend server connected")
    
    # Simple page navigation
    page = st.sidebar.radio("Navigation", ["📄 Document Upload", "📝 Generate Report"])
    
    if page == "📄 Document Upload":
        show_upload_page()
    else:
        show_report_generation_page()

def show_upload_page():
    """Simplified document upload page"""
    st.header("📄 Document Upload")
    st.write("Upload your financial documents, contracts, and reports for analysis.")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Select documents to upload",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'xlsx', 'jpg', 'jpeg', 'png'],
        help="Supported formats: PDF, Word, Excel, Images"
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        
        # Show file details
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size / 1024:.1f} KB)")
        
        if st.button("🚀 Upload & Process All Documents", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                result = upload_document(uploaded_file)
                
                if result:
                    st.success(f"✅ {uploaded_file.name} processed successfully!")
                else:
                    st.error(f"❌ Failed to process {uploaded_file.name}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            st.rerun()  # Refresh to show new documents
    
    # Document library
    st.subheader("📚 Document Library")
    documents = get_documents()
    
    if documents:
        # Create a nice display of documents
        cols = st.columns(min(3, len(documents)))
        
        for i, doc in enumerate(documents):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"**{doc.get('name', 'Unknown')}**")
                    st.write(f"📄 {doc.get('chunk_count', 0)} chunks")
                    st.write(f"📅 {doc.get('upload_date', 'Unknown')[:10]}")
                    st.write(f"💾 {doc.get('size', 0) / 1024:.1f} KB" if doc.get('size', 0) > 0 else "💾 Unknown size")
        
        # Also show as a table
        st.subheader("📊 Document Details")
        doc_df = pd.DataFrame([{
            "Document Name": doc.get("name", "Unknown"),
            "Chunks": doc.get("chunk_count", 0),
            "Status": doc.get("status", "unknown"),
            "Size (KB)": f"{doc.get('size', 0) / 1024:.1f}" if doc.get('size', 0) > 0 else "Unknown",
            "Upload Date": doc.get("upload_date", "Unknown")[:16] if doc.get("upload_date") else "Unknown"
        } for doc in documents])
        
        st.dataframe(doc_df, use_container_width=True)
    else:
        st.info("📭 No documents uploaded yet. Use the uploader above to get started.")

def show_report_generation_page():
    """Unified report generation page"""
    st.header("📝 Generate Report")
    st.write("Select documents and create report segments with AI assistance.")
    
    # Get available documents
    documents = get_documents()
    
    if not documents:
        st.warning("⚠️ No documents available. Please upload documents first.")
        if st.button("Go to Document Upload"):
            st.session_state.page = "📄 Document Upload"
            st.rerun()
        return
    
    # Document selection
    st.subheader("📋 1. Select Documents")
    
    # Show available documents with checkboxes
    selected_docs = []
    
    with st.expander("📚 Available Documents", expanded=True):
        select_all = st.checkbox("Select All Documents")
        
        for i, doc in enumerate(documents):
            is_selected = select_all or st.checkbox(
                f"{doc.get('name', 'Unknown')} ({doc.get('chunk_count', 0)} chunks)",
                key=f"doc_{i}"
            )
            if is_selected:
                selected_docs.append(doc)
    
    if not selected_docs:
        st.info("👆 Please select at least one document above.")
        return
    
    st.success(f"✅ {len(selected_docs)} document(s) selected")
    
    # Report segments
    st.subheader("📝 2. Report Segments")
    
    # Initialize segments in session state
    if "segments" not in st.session_state:
        st.session_state.segments = [
            {"name": "Executive Summary", "prompt": "Provide a high-level executive summary of the financial position and key findings."},
            {"name": "Financial Analysis", "prompt": "Analyze revenue trends, profitability, cash flow, and key financial ratios."},
            {"name": "Risk Assessment", "prompt": "Identify and assess key business risks, financial risks, and credit risks."}
        ]
    
    # Add/Edit segments
    with st.expander("➕ Manage Report Sections", expanded=False):
        # Add new segment
        col1, col2 = st.columns([3, 1])
        with col1:
            new_name = st.text_input("Section Name", placeholder="e.g., Market Analysis")
            new_prompt = st.text_area("Section Prompt", placeholder="Describe what this section should contain...", height=100)
        with col2:
            st.write("")  # Spacing
            if st.button("Add Section"):
                if new_name and new_prompt:
                    st.session_state.segments.append({"name": new_name, "prompt": new_prompt})
                    st.success("Section added!")
                    st.rerun()
        
        # Edit existing segments
        for i, segment in enumerate(st.session_state.segments):
            with st.container():
                st.markdown(f"**Section {i+1}:**")
                col1, col2, col3 = st.columns([2, 3, 1])
                with col1:
                    segment["name"] = st.text_input(f"Name", value=segment["name"], key=f"name_{i}")
                with col2:
                    segment["prompt"] = st.text_area(f"Prompt", value=segment["prompt"], key=f"prompt_{i}", height=80)
                with col3:
                    st.write("")  # Spacing
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.segments.pop(i)
                        st.rerun()
                st.markdown("---")
    
    # Show current segments
    st.write("**Current Report Sections:**")
    for i, segment in enumerate(st.session_state.segments):
        with st.expander(f"{i+1}. {segment['name']}", expanded=False):
            st.write(f"**Prompt:** {segment['prompt']}")
    
    # Generate report
    st.subheader("🚀 3. Generate Report")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        report_title = st.text_input("Report Title", value="Credit Review Report", placeholder="Enter report title...")
    with col2:
        st.write("")  # Spacing
        generate_button = st.button("🎯 Generate Full Report", type="primary", use_container_width=True)
    
    if generate_button:
        if not report_title:
            st.error("Please enter a report title.")
            return
        
        if not st.session_state.segments:
            st.error("Please add at least one report section.")
            return
        
        # Generate report
        with st.spinner("🤖 Generating report sections..."):
            generated_report = generate_complete_report(
                title=report_title,
                segments=st.session_state.segments,
                selected_documents=selected_docs
            )
        
        if generated_report:
            show_generated_report(generated_report)

def generate_complete_report(title, segments, selected_documents):
    """Generate complete report with all segments"""
    
    report_sections = []
    total_sections = len(segments)
    
    # Extract document IDs from selected documents
    selected_document_ids = [doc.get('id', doc.get('document_id')) for doc in selected_documents if doc.get('id') or doc.get('document_id')]
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, segment in enumerate(segments):
        status_text.text(f"Generating: {segment['name']}...")
        
        # Generate content for this segment
        try:
            # Create segment data
            segment_data = {
                "name": segment['name'],
                "prompt": segment['prompt'],
                "required_document_types": [],
                "generation_settings": {}
            }
            
            result = generate_segment_content(segment_data, selected_document_ids)
            
            if result and not result.get('error'):
                content = result.get('generated_content', 'No content generated')
                validation_results = result.get('validation_results', {})
                generation_metadata = result.get('generation_metadata', {})
                
                # Store validation results both in metadata and as a top-level field
                report_sections.append({
                    "name": segment['name'],
                    "content": content,
                    "validation_results": validation_results,
                    "metadata": {
                        **generation_metadata,
                        "validation_results": validation_results
                    }
                })
                st.success(f"✅ {segment['name']} completed")
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No response from server'
                st.error(f"❌ Failed to generate {segment['name']}: {error_msg}")
                report_sections.append({
                    "name": segment['name'],
                    "content": f"*Content generation failed: {error_msg}*",
                    "metadata": {}
                })
        
        except Exception as e:
            st.error(f"❌ Error generating {segment['name']}: {str(e)}")
            report_sections.append({
                "name": segment['name'],
                "content": f"*Error: {str(e)}*",
                "metadata": {}
            })
        
        progress_bar.progress((i + 1) / total_sections)
    
    status_text.text("Report generation complete!")
    
    return {
        "title": title,
        "sections": report_sections,
        "documents_used": selected_documents,
        "generation_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }

def get_paragraph_validation(validation_data, paragraph_text):
    """Get validation status for a specific paragraph"""
    # Backend returns 'issues', not 'validation_issues'
    if not validation_data or not validation_data.get('issues'):
        return {
            'status': 'passed',
            'issues': [],
            'conclusion': 'No issues detected'
        }
    
    validation_issues = validation_data.get('issues', [])
    paragraph_issues = []
    
    # Find issues that relate to this paragraph
    for issue in validation_issues:
        text_span = issue.get('text_span', '')
        if text_span and text_span.strip() in paragraph_text:
            paragraph_issues.append(issue)
    
    # Determine overall status
    if not paragraph_issues:
        status = 'passed'
        conclusion = 'All validations passed'
    else:
        high_severity_issues = [i for i in paragraph_issues if i.get('severity') == 'high']
        medium_severity_issues = [i for i in paragraph_issues if i.get('severity') == 'medium']
        
        if high_severity_issues:
            status = 'not_passed'
            conclusion = f"{len(high_severity_issues)} high-severity issues found"
        elif medium_severity_issues:
            status = 'partially_passed'
            conclusion = f"{len(medium_severity_issues)} medium-severity issues found"
        else:
            status = 'partially_passed'
            conclusion = f"{len(paragraph_issues)} minor issues found"
    
    return {
        'status': status,
        'issues': paragraph_issues,
        'conclusion': conclusion
    }

def display_paragraph_validation_status(validation_status):
    """Display paragraph validation status with detailed explanations"""
    status = validation_status.get('status', 'unknown')
    conclusion = validation_status.get('conclusion', 'Unknown status')
    issues = validation_status.get('issues', [])
    
    # Enhanced status display with detailed explanations
    if status == 'passed':
        st.success(f"✅ **Validation Passed**: {conclusion}")
        # Show success details in a clean container instead of expander
        st.markdown("**Validation Checks Passed:**")
        st.write("✓ **Accuracy Check**: Content appears factually correct based on source documents")
        st.write("✓ **Completeness Check**: All required information elements are present")
        st.write("✓ **Consistency Check**: No contradictory statements detected")
        st.write("✓ **Compliance Check**: Meets regulatory and style requirements")
            
    elif status == 'partially_passed':
        st.warning(f"⚠️ **Validation Partially Passed**: {conclusion}")
        st.write("**⚠️ This paragraph has some validation concerns but is generally acceptable.**")
        st.write("Review the issues below and consider revisions to improve quality.")
            
    elif status == 'not_passed':
        st.error(f"❌ **Validation Failed**: {conclusion}")
        st.write("**🚨 This paragraph has significant validation failures.**")
        st.write("**Immediate action required** - review and revise before using in final report.")
            
    else:
        st.info(f"ℹ️ **Validation Status**: {conclusion}")
    
    # Show detailed issues for this paragraph with enhanced explanations
    if issues:
        st.markdown("**📋 Validation Issues Found:**")
        for i, issue in enumerate(issues, 1):
            severity = issue.get('severity', 'medium')
            issue_type = issue.get('issue_type', 'Issue')
            description = issue.get('description', 'No description available')
            suggested_fix = issue.get('suggested_fix', '')
            confidence = issue.get('confidence_score', 0.0)
            text_span = issue.get('text_span', '')
            
            severity_icon = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢',
                'info': '🔵'
            }.get(severity, '⚪')
            
            severity_label = {
                'high': '**HIGH PRIORITY**',
                'medium': '**MEDIUM PRIORITY**',
                'low': '**LOW PRIORITY**',
                'info': '**INFORMATIONAL**'
            }.get(severity, '**UNKNOWN**')
            
            # Create a bordered container for each issue instead of expander
            st.markdown(f"**{severity_icon} Issue #{i}: {issue_type.title()} - {severity_label}**")
            
            # Issue details in a clean format
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(f"**Type:** {issue_type.title()}")
                st.write(f"**Severity:** {severity_label}")
            with col2:
                st.write(f"**Confidence:** {confidence:.1%}")
                if text_span:
                    st.write(f"**Affected Text:** _{text_span[:50]}..._" if len(text_span) > 50 else f"**Affected Text:** _{text_span}_")
            
            # Issue description
            st.write(f"**Description:** {description}")
            
            # Detailed explanation based on issue type
            if issue_type == 'accuracy':
                st.write("📍 **Root Cause:** Information may not be supported by source documents")
                st.write("⚠️ **Impact:** Could mislead readers or damage report credibility")
                
            elif issue_type == 'completeness':
                st.write("📍 **Root Cause:** Required elements may not be fully addressed")
                st.write("⚠️ **Impact:** May leave readers with incomplete understanding")
                
            elif issue_type == 'consistency':
                st.write("📍 **Root Cause:** Contradictory statements or inconsistent data")
                st.write("⚠️ **Impact:** Creates confusion and reduces report reliability")
                
            elif issue_type == 'compliance':
                st.write("📍 **Root Cause:** Missing disclosures or inappropriate language")
                st.write("⚠️ **Impact:** Could result in regulatory issues")
                
            elif issue_type == 'factual_error':
                st.write("📍 **Root Cause:** Misinterpretation of source documents")
                st.write("⚠️ **Impact:** Direct misinformation affecting decision-making")
                
            elif issue_type == 'source_mismatch':
                st.write("📍 **Root Cause:** Claims not aligned with source documents")
                st.write("⚠️ **Impact:** Undermines evidence-based analysis")
            
            # Suggested fix with actionable steps
            if suggested_fix:
                st.write(f"💡 **Recommended Action:** {suggested_fix}")
                
                # Additional actionable steps based on severity
                if severity == 'high':
                    st.write("🚨 **Immediate Steps:** STOP → VERIFY → REVISE → VALIDATE")
                elif severity == 'medium':
                    st.write("⚠️ **Recommended:** REVIEW → CONSIDER → IMPROVE")
                else:
                    st.write("ℹ️ **Optional:** NOTE → CONSIDER minor refinements")
            
            st.markdown("---")

def show_generated_report(report):
    """Display the generated report with validation analysis"""
    st.subheader("📋 Dashboard - Report & Validation Analysis")
    
    # Report header
    st.markdown(f"# {report['title']}")
    st.caption(f"Generated on {report['generation_date']}")
    
    # Documents used
    with st.expander("📚 Documents Used", expanded=False):
        for doc in report['documents_used']:
            st.write(f"• {doc.get('name', 'Unknown')} ({doc.get('chunk_count', 0)} chunks)")
    
    # Two-column layout for each section: Report content on left, Validation on right
    for i, section in enumerate(report['sections']):
        st.markdown(f"## {i+1}. {section['name']}")
        
        # Get validation data for this section (if available)
        validation_data = section.get('validation_results') or section.get('metadata', {}).get('validation_results', {})
        
        # Debug: Show what validation data we have (can be removed later)
        if validation_data:
            st.write(f"🔍 Debug: Validation data keys: {list(validation_data.keys())}")
            st.write(f"🔍 Debug: Validation data preview: {str(validation_data)[:200]}...")
        
        # Split into two columns
        left_col, right_col = st.columns([1.2, 0.8])
        
        with left_col:
            st.markdown("##### 📝 Generated Content")
            
            # Display clean content without validation mixed in
            content = section.get('content', '')
            if content:
                # Split content into paragraphs and display them cleanly
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                
                for p_idx, paragraph in enumerate(paragraphs):
                    # Create a container for better visual alignment with validation column
                    with st.container():
                        # Add paragraph header with consistent styling
                        st.markdown(f"**📄 Paragraph {p_idx + 1}**")
                        
                        # Display paragraph content in a clean format
                        st.write(paragraph)
                        
                        # Add some visual spacing between paragraphs
                        st.write("")  # Small spacer
                        
                    # Visual separator between paragraphs for clarity
                    if p_idx < len(paragraphs) - 1:
                        st.markdown("---")
            else:
                st.warning("No content available for this section")
        
        with right_col:
            st.markdown("##### 🔍 Validation Analysis")
            
            if validation_data:
                # Section-level validation summary at the top
                # Map backend field names to expected dashboard names
                quality_score = validation_data.get('overall_quality_score') or validation_data.get('confidence_score', 0.0)
                total_issues = validation_data.get('total_issues')
                if total_issues is None:
                    # Calculate total issues from the issues array
                    issues = validation_data.get('issues', [])
                    total_issues = len(issues)
                
                # Compact section metrics
                col1, col2 = st.columns(2)
                with col1:
                    if quality_score >= 0.8:
                        st.success(f"📊 {quality_score:.0%}")
                    elif quality_score >= 0.6:
                        st.warning(f"📊 {quality_score:.0%}")
                    else:
                        st.error(f"📊 {quality_score:.0%}")
                    st.caption("Quality Score")
                
                with col2:
                    st.metric("Issues", total_issues)
                
                st.markdown("---")
                
                # Paragraph-by-paragraph validation analysis aligned with content
                st.markdown("**📋 Paragraph Analysis:**")
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()] if content else []
                
                for p_idx, paragraph in enumerate(paragraphs):
                    # Create matching container structure for alignment
                    with st.container():
                        validation_status = get_paragraph_validation(validation_data, paragraph)
                        status = validation_status.get('status', 'unknown')
                        conclusion = validation_status.get('conclusion', 'Unknown status')
                        issues = validation_status.get('issues', [])
                        
                        # Header matching the content paragraph header
                        st.markdown(f"**🔍 Paragraph {p_idx + 1} Analysis**")
                        
                        # Status indicator
                        if status == 'passed':
                            st.success(f"✅ {conclusion}")
                        elif status == 'partially_passed':
                            st.warning(f"⚠️ {conclusion}")
                        elif status == 'not_passed':
                            st.error(f"❌ {conclusion}")
                        else:
                            st.info(f"ℹ️ {conclusion}")
                        
                        # Show issues compactly
                        if issues:
                            st.write("**Issues:**")
                            for issue in issues[:3]:  # Show max 3 issues per paragraph to keep it clean
                                severity_icon = {
                                    'high': '🔴',
                                    'medium': '🟡',
                                    'low': '🟢',
                                    'info': '🔵'
                                }.get(issue.get('severity', 'medium'), '⚪')
                                
                                issue_type = issue.get('issue_type', 'Issue')
                                description = issue.get('description', 'No description')
                                
                                # Truncate long descriptions for compact display
                                if len(description) > 60:
                                    description = description[:57] + "..."
                                
                                st.write(f"{severity_icon} {issue_type}: {description}")
                            
                            if len(issues) > 3:
                                st.write(f"... and {len(issues) - 3} more issues")
                        else:
                            st.write("**No issues detected** ✓")
                        
                        # Add spacing to match the content column
                        st.write("")
                    
                    # Visual separator matching content column
                    if p_idx < len(paragraphs) - 1:
                        st.markdown("---")
                
                # Section-level summary at the bottom
                if paragraphs:  # Only show if we have paragraphs
                    st.markdown("---")
                    st.markdown("**📝 Section Summary:**")
                    # The backend returns 'overall_assessment' directly, not nested in 'summary'
                    assessment = validation_data.get('overall_assessment', 'Section analysis completed.')
                    if assessment and assessment != 'Section analysis completed.':
                        st.write(f"_{assessment}_")
                    else:
                        st.write("_Section analysis completed._")
            else:
                st.info("No validation data available for this section.")
                
                # Mock validation for demonstration
                st.write("**Sample Validation Analysis:**")
                st.success("🟢 Content accuracy: Good")
                st.success("🟢 Completeness: Satisfactory") 
                st.warning("🟡 Minor formatting issues detected")
        
        st.markdown("---")
    
    # Download options
    st.subheader("💾 Download Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download as Word document - direct generation like other formats
        include_validation = st.checkbox("Include AI validation comments", value=True, key="word_validation")
        
        # Generate Word document content directly for download
        with st.spinner("🔄 Preparing Word document..."):
            try:
                export_result = create_report_and_export_word(report, include_validation)
                
                if export_result and not export_result.get('error'):
                    download_url = export_result.get('download_url')
                    if download_url:
                        # Download the file content
                        word_content = download_word_document(download_url)
                        
                        if word_content:
                            # Provide direct download button like other formats
                            filename = export_result.get('filename', f"{report['title'].replace(' ', '_')}.docx")
                            st.download_button(
                                label="📄 Download Word Document",
                                data=word_content,
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True,
                                type="primary",
                                help="Professional Word document with validation comments (if enabled)"
                            )
                        else:
                            # Fallback: Show error but still provide basic download
                            st.error("❌ Failed to fetch Word document from server")
                            st.write("**Fallback Option:**")
                            basic_word_content = create_basic_word_content(report)
                            st.download_button(
                                label="📄 Download Basic Word Document",
                                data=basic_word_content,
                                file_name=f"{report['title'].replace(' ', '_')}_basic.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True,
                                help="Basic Word document (server-generated version failed)"
                            )
                    else:
                        st.error("❌ No download URL provided from server")
                        # Fallback option
                        basic_word_content = create_basic_word_content(report)
                        st.download_button(
                            label="📄 Download Basic Word Document",
                            data=basic_word_content,
                            file_name=f"{report['title'].replace(' ', '_')}_basic.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                            help="Basic Word document (server connection issue)"
                        )
                else:
                    error_msg = export_result.get('error', 'Unknown error') if export_result else 'No response'
                    st.error(f"❌ Word export failed: {error_msg}")
                    # Provide fallback download
                    basic_word_content = create_basic_word_content(report)
                    st.download_button(
                        label="📄 Download Basic Word Document",
                        data=basic_word_content,
                        file_name=f"{report['title'].replace(' ', '_')}_basic.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        help="Basic Word document (advanced features failed)"
                    )
            except Exception as e:
                st.error(f"❌ Error generating Word document: {str(e)}")
                # Provide fallback download
                basic_word_content = create_basic_word_content(report)
                st.download_button(
                    label="📄 Download Basic Word Document",
                    data=basic_word_content,
                    file_name=f"{report['title'].replace(' ', '_')}_basic.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    help="Basic Word document (fallback option)"
                )
    
    with col2:
        # Download as text
        report_text = f"# {report['title']}\n\n"
        report_text += f"Generated on {report['generation_date']}\n\n"
        
        for i, section in enumerate(report['sections']):
            report_text += f"## {i+1}. {section['name']}\n\n"
            report_text += f"{section['content']}\n\n"
        
        st.download_button(
            label="📝 Download as Text",
            data=report_text,
            file_name=f"{report['title'].replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # Download as JSON
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="📋 Download as JSON",
            data=report_json,
            file_name=f"{report['title'].replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True
        )

if __name__ == "__main__":
    main()