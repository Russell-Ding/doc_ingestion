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
    initial_sidebar_state="expanded"
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

def create_report(title, description=None):
    """Create a new report"""
    try:
        data = {"title": title}
        if description:
            data["description"] = description
        response = requests.post(f"{API_BASE_URL}/reports/", json=data)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Failed to create report: {str(e)}")
        return None

def create_segment(report_id, segment_data):
    """Create a new segment"""
    try:
        segment_data["report_id"] = report_id
        response = requests.post(f"{API_BASE_URL}/segments/", json=segment_data)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Failed to create segment: {str(e)}")
        return None

def generate_segment_content(segment_id, validation_enabled=True):
    """Generate content for a segment"""
    try:
        data = {"validation_enabled": validation_enabled}
        response = requests.post(f"{API_BASE_URL}/segments/{segment_id}/generate", json=data)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Failed to generate content: {str(e)}")
        return None

def generate_report(report_id, segment_ids, validation_enabled=True):
    """Generate entire report"""
    try:
        data = {
            "segments": segment_ids,
            "validation_enabled": validation_enabled
        }
        response = requests.post(f"{API_BASE_URL}/reports/{report_id}/generate", json=data)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Failed to generate report: {str(e)}")
        return None

def export_report(report_id, format_type="word", include_validations=True):
    """Export report"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/reports/{report_id}/export/{format_type}",
            params={"include_validations": include_validations}
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Failed to export report: {str(e)}")
        return None

# Initialize session state
if "report_id" not in st.session_state:
    st.session_state.report_id = None
if "segments" not in st.session_state:
    st.session_state.segments = []
if "documents" not in st.session_state:
    st.session_state.documents = []

# Main app
st.title("🏦 Credit Review Report Generation System")
st.markdown("AI-powered document analysis for credit risk managers")

# Check backend status
backend_status = check_backend_health()
if not backend_status:
    st.error("🔴 Backend server is not running. Please start the backend first:")
    st.code("cd backend && uvicorn app.main:app --reload")
    st.stop()
else:
    st.success("🟢 Backend server is running")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "📄 Document Upload", 
    "📝 Report Builder", 
    "🤖 Generate Content", 
    "📊 View Results",
    "📁 Export Report"
])

# Document Upload Page
if page == "📄 Document Upload":
    st.header("📄 Document Upload")
    
    # Upload section
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, Word, Excel, Images)",
        type=['pdf', 'docx', 'xlsx', 'xls', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if st.button(f"Process {uploaded_file.name}", key=f"upload_{uploaded_file.name}"):
                with st.spinner(f"Uploading and processing {uploaded_file.name}..."):
                    result = upload_document(uploaded_file, uploaded_file.name)
                    if result:
                        st.success(f"✅ {uploaded_file.name} processed successfully!")
                        st.json(result)
                        # Refresh documents list
                        st.session_state.documents = get_documents()
                    else:
                        st.error(f"❌ Failed to process {uploaded_file.name}")
    
    # Show uploaded documents
    st.subheader("Uploaded Documents")
    documents = get_documents()
    st.session_state.documents = documents
    
    if documents:
        doc_data = []
        for doc in documents:
            doc_data.append({
                "Name": doc.get("name", "Unknown"),
                "Chunks": doc.get("chunk_count", 0),
                "Status": doc.get("status", "unknown"),
                "Size": f"{doc.get('size', 0) / 1024:.1f} KB" if doc.get('size', 0) > 0 else "Unknown",
                "Upload Date": doc.get("upload_date", "Unknown")
            })
        
        df = pd.DataFrame(doc_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No documents uploaded yet. Upload some documents to get started!")

# Report Builder Page
elif page == "📝 Report Builder":
    st.header("📝 Report Builder")
    
    # Create or select report
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not st.session_state.report_id:
            st.subheader("Create New Report")
            report_title = st.text_input("Report Title", "Credit Review Report")
            report_description = st.text_area("Description (optional)", "Comprehensive credit analysis report")
            
            if st.button("Create Report", type="primary"):
                with st.spinner("Creating report..."):
                    report_data = create_report(report_title, report_description)
                    if report_data:
                        st.session_state.report_id = report_data["id"]
                        st.success(f"✅ Report created: {report_data['title']}")
                        st.rerun()
        else:
            st.success(f"📊 Working on Report ID: {st.session_state.report_id}")
            if st.button("Start New Report"):
                st.session_state.report_id = None
                st.session_state.segments = []
                st.rerun()
    
    if st.session_state.report_id:
        st.divider()
        st.subheader("Report Segments")
        
        # Add new segment
        with st.expander("➕ Add New Segment", expanded=len(st.session_state.segments) == 0):
            segment_name = st.text_input("Segment Name", "Financial Summary")
            
            # Template selection
            template_options = {
                "Custom": "Write your own prompt",
                "Financial Summary": "Analyze financial performance including revenue, profitability, and key ratios",
                "Risk Assessment": "Evaluate credit risk factors and provide risk rating",
                "Cash Flow Analysis": "Analyze cash flow patterns and sustainability",
                "Industry Analysis": "Provide industry and market analysis",
                "Management Assessment": "Evaluate management team quality"
            }
            
            selected_template = st.selectbox("Template", list(template_options.keys()))
            
            if selected_template == "Custom":
                segment_prompt = st.text_area("Prompt", "Enter your custom analysis requirements...")
            else:
                default_prompts = {
                    "Financial Summary": "Please provide a comprehensive financial summary based on the available financial documents. Include key metrics such as revenue, profitability, cash flow, and financial ratios. Highlight any significant trends or concerns.",
                    "Risk Assessment": "Analyze the credit risk factors based on the provided documentation. Evaluate the borrower's creditworthiness, including financial stability, industry risks, market position, and management quality. Provide a risk rating and key concerns.",
                    "Cash Flow Analysis": "Analyze the cash flow patterns from the financial statements and projections. Evaluate the quality and sustainability of cash flows, seasonal variations, and cash flow coverage ratios.",
                    "Industry Analysis": "Provide an analysis of the industry in which the borrower operates. Include market conditions, competitive landscape, regulatory environment, and industry-specific risks and opportunities.",
                    "Management Assessment": "Evaluate the management team's experience, track record, and ability to execute business plans. Consider leadership changes, key person risks, and management depth."
                }
                segment_prompt = st.text_area("Prompt", default_prompts.get(selected_template, ""))
            
            # Document types
            doc_types = st.multiselect(
                "Required Document Types",
                ["financial_statements", "cash_flow", "contracts", "business_plan", "tax_returns", "bank_statements"],
                default=["financial_statements"]
            )
            
            # Settings
            col1, col2 = st.columns(2)
            with col1:
                max_tokens = st.slider("Max Length (tokens)", 500, 4000, 2000)
                include_tables = st.checkbox("Include Tables", True)
            with col2:
                temperature = st.slider("Creativity", 0.1, 1.0, 0.7)
                validation_enabled = st.checkbox("Enable Validation", True)
            
            if st.button("Add Segment", type="primary"):
                segment_data = {
                    "name": segment_name,
                    "description": template_options.get(selected_template, ""),
                    "prompt": segment_prompt,
                    "order_index": len(st.session_state.segments),
                    "required_document_types": doc_types,
                    "generation_settings": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "include_tables": include_tables,
                        "validation_enabled": validation_enabled
                    }
                }
                
                result = create_segment(st.session_state.report_id, segment_data)
                if result:
                    st.session_state.segments.append(result)
                    st.success(f"✅ Added segment: {segment_name}")
                    st.rerun()
        
        # Show existing segments
        if st.session_state.segments:
            st.subheader("Current Segments")
            for i, segment in enumerate(st.session_state.segments):
                with st.expander(f"{i+1}. {segment['name']}", expanded=False):
                    st.write(f"**Description:** {segment.get('description', 'N/A')}")
                    st.write(f"**Status:** {segment.get('content_status', 'pending')}")
                    st.text_area("Prompt:", segment['prompt'], disabled=True, key=f"prompt_{i}")
                    if segment.get('required_document_types'):
                        st.write(f"**Required Documents:** {', '.join(segment['required_document_types'])}")
        else:
            st.info("No segments added yet. Add your first segment above!")

# Generate Content Page
elif page == "🤖 Generate Content":
    st.header("🤖 Generate Content")
    
    if not st.session_state.report_id:
        st.warning("Please create a report first in the Report Builder page.")
    elif not st.session_state.segments:
        st.warning("Please add segments to your report first.")
    else:
        st.write(f"**Report ID:** {st.session_state.report_id}")
        st.write(f"**Segments:** {len(st.session_state.segments)}")
        
        # Documents check
        if not st.session_state.documents:
            st.warning("⚠️ No documents uploaded. Upload documents first for better content generation.")
        else:
            st.success(f"✅ {len(st.session_state.documents)} documents available")
        
        st.divider()
        
        # Generate individual segments
        st.subheader("Generate Individual Segments")
        for i, segment in enumerate(st.session_state.segments):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{i+1}. {segment['name']}**")
                st.write(f"Status: {segment.get('content_status', 'pending')}")
            
            with col2:
                if st.button(f"Generate", key=f"gen_{i}"):
                    with st.spinner(f"Generating content for {segment['name']}..."):
                        result = generate_segment_content(segment['id'])
                        if result:
                            st.success("✅ Content generated!")
                            # Update segment in session state
                            st.session_state.segments[i]['content_status'] = 'completed'
                            st.session_state.segments[i]['generated_content'] = result.get('generated_content', '')
                        else:
                            st.error("❌ Generation failed")
            
            with col3:
                if segment.get('generated_content'):
                    if st.button(f"View", key=f"view_{i}"):
                        st.text_area("Generated Content:", segment['generated_content'], height=200)
        
        st.divider()
        
        # Generate entire report
        st.subheader("Generate Entire Report")
        col1, col2 = st.columns(2)
        
        with col1:
            validation_enabled = st.checkbox("Enable Content Validation", True, key="report_validation")
        
        with col2:
            if st.button("🚀 Generate Full Report", type="primary"):
                segment_ids = [seg['id'] for seg in st.session_state.segments]
                
                with st.spinner("Generating full report... This may take several minutes."):
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(len(segment_ids)):
                        status_text.text(f"Processing segment {i+1}/{len(segment_ids)}: {st.session_state.segments[i]['name']}")
                        progress_bar.progress((i + 1) / len(segment_ids))
                        
                        # Generate segment content
                        result = generate_segment_content(segment_ids[i], validation_enabled)
                        if result:
                            st.session_state.segments[i]['content_status'] = 'completed'
                            st.session_state.segments[i]['generated_content'] = result.get('generated_content', '')
                        
                        time.sleep(1)  # Small delay for demo purposes
                    
                    status_text.text("Report generation completed!")
                    st.success("🎉 Full report generated successfully!")

# View Results Page
elif page == "📊 View Results":
    st.header("📊 View Results")
    
    if not st.session_state.segments:
        st.info("No segments to display. Create and generate content first.")
    else:
        # Report summary
        completed_segments = [s for s in st.session_state.segments if s.get('content_status') == 'completed']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Segments", len(st.session_state.segments))
        with col2:
            st.metric("Completed", len(completed_segments))
        with col3:
            completion_rate = len(completed_segments) / len(st.session_state.segments) * 100 if st.session_state.segments else 0
            st.metric("Completion Rate", f"{completion_rate:.0f}%")
        
        st.divider()
        
        # Show generated content
        for i, segment in enumerate(st.session_state.segments):
            st.subheader(f"{i+1}. {segment['name']}")
            
            status = segment.get('content_status', 'pending')
            if status == 'completed':
                st.success("✅ Completed")
                content = segment.get('generated_content', '')
                if content:
                    st.text_area("Generated Content:", content, height=300, key=f"content_{i}")
                else:
                    st.warning("No content available")
            elif status == 'generating':
                st.info("⏳ Generating...")
            elif status == 'error':
                st.error("❌ Error occurred")
            else:
                st.warning("⏸ Pending generation")
            
            st.divider()

# Export Report Page
elif page == "📁 Export Report":
    st.header("📁 Export Report")
    
    if not st.session_state.report_id:
        st.warning("Please create a report first.")
    else:
        completed_segments = [s for s in st.session_state.segments if s.get('content_status') == 'completed']
        
        if not completed_segments:
            st.warning("No completed segments to export. Generate content first.")
        else:
            st.success(f"Ready to export {len(completed_segments)} completed segments")
            
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox("Export Format", ["word", "pdf"])
                include_validations = st.checkbox("Include Validation Comments", True)
            
            with col2:
                if st.button("📄 Export Report", type="primary"):
                    with st.spinner(f"Exporting report as {export_format.upper()}..."):
                        result = export_report(
                            st.session_state.report_id, 
                            export_format, 
                            include_validations
                        )
                        
                        if result:
                            st.success(f"✅ Report exported successfully!")
                            st.json(result)
                            
                            # Show download link
                            if result.get('download_url'):
                                st.markdown(f"📥 [Download Report]({result['download_url']})")
                        else:
                            st.error("❌ Export failed")
            
            # Report preview
            st.divider()
            st.subheader("Report Preview")
            
            for i, segment in enumerate(completed_segments):
                st.write(f"### {i+1}. {segment['name']}")
                content = segment.get('generated_content', '')
                if content:
                    # Truncate for preview
                    preview = content[:500] + "..." if len(content) > 500 else content
                    st.write(preview)
                st.write("---")

# Footer
st.sidebar.divider()
st.sidebar.markdown("### System Status")
if backend_status:
    st.sidebar.success("🟢 Backend Online")
else:
    st.sidebar.error("🔴 Backend Offline")

st.sidebar.markdown("### Quick Stats")
st.sidebar.metric("Documents", len(st.session_state.documents))
st.sidebar.metric("Segments", len(st.session_state.segments))

if st.session_state.segments:
    completed = len([s for s in st.session_state.segments if s.get('content_status') == 'completed'])
    st.sidebar.metric("Completed", f"{completed}/{len(st.session_state.segments)}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Credit Review System v1.0**")
st.sidebar.markdown("AI-powered document analysis")