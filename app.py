import streamlit as st
import pandas as pd
import openai
import json
import os
from io import StringIO

# Configure the page
st.set_page_config(
    page_title="AI File Mapper",
    page_icon="🔄",
    layout="wide"
)

# Initialize OpenAI client
@st.cache_resource
def init_openai():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OpenAI API key not found! Please add it to .streamlit/secrets.toml or set OPENAI_API_KEY environment variable")
        st.stop()
    return openai.OpenAI(api_key=api_key)

try:
    client = init_openai()
except:
    st.error("Please configure your OpenAI API key to continue")
    st.stop()

# Define standard output format
STANDARD_FORMAT = {
    "first_name": "First Name",
    "last_name": "Last Name", 
    "email": "Email Address",
    "phone": "Phone Number",
    "company": "Company Name",
    "job_title": "Job Title",
    "address": "Street Address",
    "city": "City",
    "state": "State/Province",
    "zip_code": "ZIP/Postal Code",
    "country": "Country"
}

def analyze_file_with_ai(df, standard_format):
    """Use AI to suggest initial mappings and ask clarifying questions"""
    
    # Get sample data for context
    sample_data = df.head(3).to_string()
    column_names = list(df.columns)
    
    prompt = f"""
    I have a CSV file with these columns: {column_names}
    
    Here's a sample of the data:
    {sample_data}
    
    I need to map these columns to this standard format: {list(standard_format.keys())}
    
    Please:
    1. Suggest the most likely mappings based on column names and sample data
    2. Identify any ambiguous columns that need clarification
    3. Ask specific questions about unclear mappings
    
    Respond in JSON format:
    {{
        "suggested_mappings": {{"standard_field": "input_column"}},
        "questions": ["Question 1 about ambiguous field X", "Question 2 about field Y"],
        "confidence": {{"input_column": 0.9}}
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.error(f"AI analysis failed: {e}")
        return {"suggested_mappings": {}, "questions": [], "confidence": {}}

def chat_with_ai(message, context):
    """Handle conversational questions about mappings"""
    
    prompt = f"""
    Context: {context}
    
    User question: {message}
    
    Please provide a helpful response about the file mapping process. 
    Be specific and actionable.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error: {e}"

def apply_mapping(df, mapping):
    """Apply the mapping to create standardized output"""
    
    mapped_df = pd.DataFrame()
    
    for standard_field in STANDARD_FORMAT.keys():
        input_column = mapping.get(standard_field)
        if input_column and input_column in df.columns:
            mapped_df[standard_field] = df[input_column]
        else:
            mapped_df[standard_field] = ""
    
    return mapped_df

def main():
    st.title("🔄 AI-Powered File Mapper")
    st.markdown("Upload a delimited file and let AI help you map it to a standard format!")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'ai_analysis' not in st.session_state:
        st.session_state.ai_analysis = None
    if 'mapping' not in st.session_state:
        st.session_state.mapping = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for standard format reference
    with st.sidebar:
        st.header("📋 Standard Format")
        st.markdown("**Target Fields:**")
        for field, description in STANDARD_FORMAT.items():
            st.markdown(f"• `{field}`: {description}")
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Start")
        st.markdown("1. Upload your CSV file")
        st.markdown("2. Click 'Analyze with AI'")
        st.markdown("3. Review suggested mappings")
        st.markdown("4. Chat with AI for help")
        st.markdown("5. Download mapped file")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload File")
        
        uploaded_file = st.file_uploader(
            "Choose a delimited file",
            type=['csv', 'txt'],
            help="Upload a CSV, TSV, or other delimited file"
        )
        
        if uploaded_file:
            # File delimiter selection
            delimiter = st.selectbox(
                "Select delimiter:",
                [",", "\t", ";", "|"],
                format_func=lambda x: {
                    ",": "Comma (,)", 
                    "\t": "Tab", 
                    ";": "Semicolon (;)", 
                    "|": "Pipe (|)"
                }[x]
            )
            
            try:
                # Read the file
                if delimiter == "\t":
                    df = pd.read_csv(uploaded_file, sep='\t')
                else:
                    df = pd.read_csv(uploaded_file, sep=delimiter)
                    
                st.session_state.df = df
                
                # Show file preview
                st.subheader("📊 File Preview")
                st.dataframe(df.head(), use_container_width=True)
                st.info(f"📈 File contains **{len(df):,}** rows and **{len(df.columns)}** columns")
                
                # AI Analysis button
                if st.button("🤖 Analyze with AI", type="primary", use_container_width=True):
                    with st.spinner("🧠 AI is analyzing your file..."):
                        st.session_state.ai_analysis = analyze_file_with_ai(df, STANDARD_FORMAT)
                        
                        # Initialize mapping with AI suggestions
                        if st.session_state.ai_analysis.get('suggested_mappings'):
                            st.session_state.mapping = st.session_state.ai_analysis['suggested_mappings']
                        
                        st.success("✅ AI analysis complete!")
                        st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                st.info("💡 Try selecting a different delimiter or check your file format")
    
    with col2:
        st.header("🔗 Field Mapping")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Show AI questions if available
            if st.session_state.ai_analysis and st.session_state.ai_analysis.get('questions'):
                st.subheader("❓ AI Questions")
                for i, question in enumerate(st.session_state.ai_analysis['questions']):
                    st.warning(f"**Q{i+1}:** {question}")
            
            # Manual mapping interface
            st.subheader("🎯 Configure Mappings")
            
            input_columns = [""] + list(df.columns)
            
            for standard_field, description in STANDARD_FORMAT.items():
                col_a, col_b = st.columns([3, 1])
                
                with col_a:
                    # Get current mapping or AI suggestion
                    current_value = st.session_state.mapping.get(standard_field, "")
                    if current_value not in input_columns:
                        current_value = ""
                    
                    selected = st.selectbox(
                        f"`{standard_field}` - {description}",
                        input_columns,
                        index=input_columns.index(current_value) if current_value in input_columns else 0,
                        key=f"mapping_{standard_field}"
                    )
                    
                    if selected:
                        st.session_state.mapping[standard_field] = selected
                    elif standard_field in st.session_state.mapping:
                        del st.session_state.mapping[standard_field]
                
                with col_b:
                    # Show confidence if available
                    if (st.session_state.ai_analysis and 
                        st.session_state.ai_analysis.get('confidence') and 
                        selected in st.session_state.ai_analysis['confidence']):
                        
                        confidence = st.session_state.ai_analysis['confidence'][selected]
                        st.metric("AI Confidence", f"{confidence:.0%}")
    
    # Chat interface
    if st.session_state.df is not None:
        st.header("💬 Ask AI About Mappings")
        
        # Display chat history
        if st.session_state.chat_history:
            with st.container():
                for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
                    st.markdown(f"**🙋 You:** {user_msg}")
                    st.markdown(f"**🤖 AI:** {ai_msg}")
                    if i < len(st.session_state.chat_history) - 1:
                        st.divider()
        
        # Chat input
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                user_question = st.text_input(
                    "Ask about field mappings:",
                    placeholder="e.g., 'What should I do with the Phone1 and Phone2 columns?'",
                    key="chat_input"
                )
            with col2:
                send_button = st.button("Send", type="secondary", use_container_width=True)
        
        if (user_question and send_button) or (user_question and st.session_state.get('send_pressed')):
            if 'send_pressed' in st.session_state:
                del st.session_state['send_pressed']
                
            context = {
                "columns": list(st.session_state.df.columns),
                "current_mapping": st.session_state.mapping,
                "standard_format": STANDARD_FORMAT,
                "sample_data": st.session_state.df.head(2).to_dict()
            }
            
            with st.spinner("🤖 AI is thinking..."):
                ai_response = chat_with_ai(user_question, str(context))
                st.session_state.chat_history.append((user_question, ai_response))
            
            st.rerun()
    
    # Preview and Download
    if st.session_state.df is not None and st.session_state.mapping:
        st.header("👀 Preview & Download")
        
        # Apply mapping
        mapped_df = apply_mapping(st.session_state.df, st.session_state.mapping)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Mapping Summary")
            mapped_count = sum(1 for v in st.session_state.mapping.values() if v)
            st.metric("Mapped Fields", f"{mapped_count}/{len(STANDARD_FORMAT)}")
            
            for standard_field, input_column in st.session_state.mapping.items():
                if input_column:
                    st.success(f"✅ `{standard_field}` ← `{input_column}`")
        
        with col2:
            st.subheader("⚠️ Unmapped Fields")
            unmapped = [col for col in st.session_state.df.columns 
                       if col not in st.session_state.mapping.values()]
            
            if unmapped:
                for col in unmapped:
                    st.warning(f"❌ `{col}` (not mapped)")
            else:
                st.success("🎉 All input fields are mapped!")
        
        # Show preview
        st.subheader("📊 Output Preview")
        st.dataframe(mapped_df.head(10), use_container_width=True)
        
        # Download button
        csv_data = mapped_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Mapped File",
            data=csv_data,
            file_name="mapped_data.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
        
        st.success(f"🎯 Ready to download {len(mapped_df):,} mapped records!")

if __name__ == "__main__":
    main()
