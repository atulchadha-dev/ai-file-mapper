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
    """Use AI to suggest initial mappings and identify unclear fields"""
    
    # Get sample data for context
    sample_data = df.head(3).to_string()
    column_names = list(df.columns)
    
    prompt = f"""
    I have a CSV file with these columns: {column_names}
    
    Here's a sample of the data:
    {sample_data}
    
    I need to map these columns to this standard format: {list(standard_format.keys())}
    Standard format descriptions: {standard_format}
    
    Please:
    1. Suggest mappings for columns you're confident about
    2. Identify columns that are unclear or ambiguous
    3. For unclear columns, DO NOT guess - mark them as needing clarification
    
    Respond in JSON format:
    {{
        "confident_mappings": {{"standard_field": "input_column"}},
        "unclear_columns": ["column1", "column2"],
        "confidence_scores": {{"input_column": 0.95}}
    }}
    
    Only include mappings you're very confident about (>80% sure).
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.error(f"AI analysis failed: {e}")
        return {"confident_mappings": {}, "unclear_columns": [], "confidence_scores": {}}

def ask_about_unclear_column(column_name, sample_values, standard_format):
    """Generate a smart question about an unclear column"""
    
    # Get a few sample values to provide context
    sample_text = ", ".join([str(v) for v in sample_values[:3] if str(v).strip()])
    
    prompt = f"""
    I have a column named "{column_name}" with sample values: {sample_text}
    
    I need to map it to one of these standard fields: {list(standard_format.keys())}
    Standard field descriptions: {standard_format}
    
    Generate a clear, concise question to ask the user what this column contains.
    The question should help determine which standard field it maps to.
    
    Format: Just return the question text, nothing else.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"What type of data does the column '{column_name}' contain?"

def interpret_user_response(user_response, column_name, standard_format):
    """Interpret user's response to determine which standard field to map to"""
    
    prompt = f"""
    The user was asked about column "{column_name}" and responded: "{user_response}"
    
    Based on their response, which of these standard fields should it map to?
    Available fields: {list(standard_format.keys())}
    Field descriptions: {standard_format}
    
    Respond with ONLY the field name (e.g., "company" or "first_name") or "NONE" if unclear.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Validate the response
        if result in standard_format.keys():
            return result
        elif result == "none":
            return None
        else:
            # Try to find a partial match
            for field in standard_format.keys():
                if field in result or result in field:
                    return field
            return None
            
    except Exception as e:
        return None

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
    if 'unclear_columns' not in st.session_state:
        st.session_state.unclear_columns = []
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'ai_questions_complete' not in st.session_state:
        st.session_state.ai_questions_complete = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'current_column' not in st.session_state:
        st.session_state.current_column = ""
    
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
        st.markdown("3. Answer AI questions about unclear fields")
        st.markdown("4. Review and adjust mappings")
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
                        
                        # Initialize mapping with confident AI suggestions
                        if st.session_state.ai_analysis.get('confident_mappings'):
                            st.session_state.mapping = st.session_state.ai_analysis['confident_mappings']
                        
                        # Set up unclear columns for questioning
                        st.session_state.unclear_columns = st.session_state.ai_analysis.get('unclear_columns', [])
                        st.session_state.current_question_index = 0
                        st.session_state.ai_questions_complete = len(st.session_state.unclear_columns) == 0
                        
                        st.success("✅ AI analysis complete!")
                        st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                st.info("💡 Try selecting a different delimiter or check your file format")
    
    with col2:
        st.header("🤖 AI Assistant")
        
        if st.session_state.df is not None and st.session_state.ai_analysis:
            
            # Show confident mappings first
            if st.session_state.ai_analysis.get('confident_mappings'):
                st.subheader("✅ AI Auto-Mapped Fields")
                for standard_field, input_column in st.session_state.ai_analysis['confident_mappings'].items():
                    confidence = st.session_state.ai_analysis.get('confidence_scores', {}).get(input_column, 0)
                    st.success(f"**{standard_field}** ← `{input_column}` ({confidence:.0%} confidence)")
            
            # AI Questioning Section
            if not st.session_state.ai_questions_complete and st.session_state.unclear_columns:
                st.subheader("❓ AI Questions")
                
                current_idx = st.session_state.current_question_index
                if current_idx < len(st.session_state.unclear_columns):
                    current_col = st.session_state.unclear_columns[current_idx]
                    
                    # Generate question if not already generated
                    if st.session_state.current_column != current_col:
                        st.session_state.current_column = current_col
                        sample_values = st.session_state.df[current_col].dropna().head(3).tolist()
                        st.session_state.current_question = ask_about_unclear_column(
                            current_col, sample_values, STANDARD_FORMAT
                        )
                    
                    # Show progress
                    st.progress((current_idx) / len(st.session_state.unclear_columns))
                    st.caption(f"Question {current_idx + 1} of {len(st.session_state.unclear_columns)}")
                    
                    # Show the question
                    st.info(f"**Column:** `{current_col}`")
                    st.markdown(f"**AI Question:** {st.session_state.current_question}")
                    
                    # Show sample data
                    sample_data = st.session_state.df[current_col].dropna().head(5).tolist()
                    st.caption(f"**Sample values:** {', '.join([str(v) for v in sample_data])}")
                    
                    # User response input
                    user_response = st.text_input(
                        "Your answer:",
                        placeholder="e.g., 'This column contains company names'",
                        key=f"response_{current_idx}"
                    )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("📝 Submit Answer", disabled=not user_response.strip()):
                            with st.spinner("🤖 Processing your answer..."):
                                # Interpret the user's response
                                mapped_field = interpret_user_response(
                                    user_response, current_col, STANDARD_FORMAT
                                )
                                
                                if mapped_field:
                                    st.session_state.mapping[mapped_field] = current_col
                                    st.success(f"✅ Mapped `{current_col}` → **{mapped_field}**")
                                else:
                                    st.warning(f"⚠️ Couldn't map `{current_col}` - you can set it manually below")
                                
                                # Move to next question
                                st.session_state.current_question_index += 1
                                
                                # Check if we're done
                                if st.session_state.current_question_index >= len(st.session_state.unclear_columns):
                                    st.session_state.ai_questions_complete = True
                                    st.balloons()
                                    st.success("🎉 All questions completed! Review your mappings below.")
                                
                                st.rerun()
                    
                    with col_b:
                        if st.button("⏭️ Skip This Column"):
                            st.session_state.current_question_index += 1
                            if st.session_state.current_question_index >= len(st.session_state.unclear_columns):
                                st.session_state.ai_questions_complete = True
                            st.rerun()
                
            elif st.session_state.ai_questions_complete:
                st.success("🎉 AI questioning complete!")
                if st.button("🔄 Restart Questions"):
                    st.session_state.current_question_index = 0
                    st.session_state.ai_questions_complete = False
                    st.rerun()
    
    # Manual Mapping Interface (always visible after AI analysis)
    if st.session_state.df is not None and st.session_state.ai_analysis:
        st.header("🔗 Review & Adjust Mappings")
        
        input_columns = [""] + list(st.session_state.df.columns)
        
        # Show current mappings in a nice format
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Field Mappings")
            
            for standard_field, description in STANDARD_FORMAT.items():
                current_value = st.session_state.mapping.get(standard_field, "")
                if current_value not in input_columns:
                    current_value = ""
                
                selected = st.selectbox(
                    f"`{standard_field}` - {description}",
                    input_columns,
                    index=input_columns.index(current_value) if current_value in input_columns else 0,
                    key=f"manual_mapping_{standard_field}"
                )
                
                if selected:
                    st.session_state.mapping[standard_field] = selected
                elif standard_field in st.session_state.mapping:
                    del st.session_state.mapping[standard_field]
        
        with col2:
            st.subheader("📊 Mapping Status")
            
            mapped_fields = [f for f, col in st.session_state.mapping.items() if col]
            unmapped_input_cols = [col for col in st.session_state.df.columns 
                                 if col not in st.session_state.mapping.values()]
            
            st.metric("Mapped Fields", f"{len(mapped_fields)}/{len(STANDARD_FORMAT)}")
            st.metric("Unmapped Input Columns", len(unmapped_input_cols))
            
            if mapped_fields:
                st.markdown("**✅ Mapped:**")
                for field in mapped_fields:
                    st.markdown(f"• {field} ← `{st.session_state.mapping[field]}`")
            
            if unmapped_input_cols:
                st.markdown("**⚠️ Unmapped Input Columns:**")
                for col in unmapped_input_cols:
                    st.markdown(f"• `{col}`")
    
    # Preview and Download
    if st.session_state.df is not None and st.session_state.mapping:
        st.header("👀 Preview & Download")
        
        # Apply mapping
        mapped_df = apply_mapping(st.session_state.df, st.session_state.mapping)
        
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
