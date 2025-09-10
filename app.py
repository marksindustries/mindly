import os
import streamlit as st
from dotenv import load_dotenv

from rag_chain import (
    save_upload_and_index, get_retriever, _get_relevant_documents, 
    check_course_status, clear_all_cache, list_all_courses,
    delete_course, get_chromadb_info, test_chromadb_connection,
    get_embedding_info
)
from generators import (
    rag_answer,
    summary_from_context,
    interview_qs_from_context,
    quiz_from_context,
)

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from prompts import SYSTEM_PROMPT


# Load environment variables
load_dotenv()
APP_NAME = os.getenv("APP_NAME", "AI Study Companion")
LOGO_URL = os.getenv("LOGO_URL", "")

# Streamlit page config
st.set_page_config(page_title=APP_NAME, page_icon="🎓", layout="wide")

# Display custom logo.png at the top
st.image("logo.png", width=120)  # Adjust width as needed
st.title(APP_NAME)

# Performance indicator
if st.sidebar.button("🧹 Clear All Cache", help="Clear cache to free memory"):
    clear_all_cache()
    st.success("Cache cleared! Page will refresh.")
    st.rerun()

def init_chat_state():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

def add_chat_message(role, content):
    st.session_state.chat_messages.append({"role": role, "content": content})

@st.cache_data(ttl=300)  # Cache for 5 minutes
def build_context_block(course_name: str, user_query: str, k: int = 5):
    """Retrieve top-k chunks for the query and format a short context block."""
    try:
        docs = _get_relevant_documents(course_name, user_query, k)
        if not docs:
            return "", []
        
        formatted = []
        seen = set()
        for d in docs:
            title = d.metadata.get("source", "doc")
            key = (title, d.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            formatted.append(f"[Source: {title}]\n{d.page_content}")
        return "\n\n".join(formatted), docs
    except Exception as e:
        st.error(f"Error building context: {e}")
        return "", []

# Enhanced Sidebar with performance monitoring
with st.sidebar:
    # App branding section
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; border-bottom: 2px solid #f0f2f6;">
        <h2 style="color: #1f77b4; margin: 0;">Mindly</h2>
        <p style="color: #666; margin: 5px 0 0 0; font-size: 14px;">AI-Powered Learning Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Course Configuration Section
    st.markdown("### 📚 Course Setup")
    
    # Course name with validation
    course = st.text_input(
        "Course Name",
        value="Demo Course",
        placeholder="e.g., Physics 101, Data Structures",
        help="Enter your course name - this helps organize your materials",
        key="course_name"
    )
    
    # Course status check with ChromaDB Cloud
    if course and course != "Demo Course":
        status = check_course_status(course)
        
        if "error" in status:
            st.error(f"⚠️ **{course}** - Database error: {status['error']}")
        elif status["is_ready"]:
            st.success(f"✅ **{course}** ({status['document_count']} docs in cloud)")
        else:
            st.warning(f"⚠️ **{course}** - No materials indexed yet")
    else:
        st.info("💡 Enter your course name to get started")
    
    # Quick course actions
    if course:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄", help="Reset course", key="reset_course"):
                st.session_state.course_name = "Demo Course"
                st.rerun()
        with col2:
            if st.button("📊", help="Check course status", key="check_files"):
                status = check_course_status(course)
                st.json(status)
    
    st.markdown("---")
    
    # Navigation Section with icons and descriptions
    st.markdown("### 🧭 Navigation")
    
    # Mode selection with enhanced options
    mode_options = {
        "📤 Setup": "Admin (Upload/Index)",
        "💬 Chat": "Chat", 
        "📝 Quiz": "Quiz",
        "📋 Summary": "Summary",
        "🎤 Interview": "Interview Qs"
    }
    
    # Custom radio with descriptions
    selected_display = st.radio(
        "Choose Mode:",
        options=list(mode_options.keys()),
        help="Select what you want to do with your course materials"
    )
    
    section = mode_options[selected_display]
    
    # Mode descriptions
    mode_descriptions = {
        "📤 Setup": "Upload and process your course files",
        "💬 Chat": "Ask questions about your materials", 
        "📝 Quiz": "Generate practice quizzes",
        "📋 Summary": "Create study summaries",
        "🎤 Interview": "Practice exam questions"
    }
    
    st.caption(f"🔍 {mode_descriptions[selected_display]}")
    
    st.markdown("---")
    
    # Quick Stats Section
    st.markdown("### 📈 Quick Stats")
    
    # Session statistics
    if hasattr(st.session_state, 'chat_messages') and st.session_state.chat_messages:
        chat_count = len([m for m in st.session_state.chat_messages if m["role"] == "user"])
        st.metric("💬 Questions Asked", chat_count)
    
    if hasattr(st.session_state, 'quiz_data') and st.session_state.quiz_data:
        st.metric("📝 Quiz Questions", len(st.session_state.quiz_data))
    
    # Course material indicator (using cached status check)
    if course and course != "Demo Course":
        status = check_course_status(course)
        if status["is_ready"]:
            st.success(f"📚 {status['document_count']} Documents Ready")
        else:
            st.warning("📤 Upload Materials First")
    
    st.markdown("---")
    
    # Performance Section with ChromaDB info
    st.markdown("### ⚡ Performance")
    
    # ChromaDB connection status
    chromadb_info = get_chromadb_info()
    if chromadb_info["connected"]:
        if chromadb_info["using_cloud"]:
            st.success(f"☁️ ChromaDB Cloud ({chromadb_info['total_collections']} collections)")
        else:
            st.info(f"💾 Local ChromaDB ({chromadb_info['total_collections']} collections)")
    else:
        st.error("❌ ChromaDB Connection Failed")
    
    # Embedding model info
    embedding_info = get_embedding_info()
    st.caption(f"🤖 Model: {embedding_info['model_name']}")
    st.caption(f"📐 Dimension: {embedding_info['embedding_dimension']}")
    
    # Cache status indicator
    cache_info = st.session_state.get('_resource_cache', {})
    if cache_info:
        st.success(f"🚀 Cache Active ({len(cache_info)} items)")
    else:
        st.info("🔄 Building cache...")
    
    # Database management
    if st.button("🔧 Test Connection", help="Test ChromaDB connection"):
        success, message = test_chromadb_connection()
        if success:
            st.success(message)
        else:
            st.error(message)
    
    # Quick Actions Section
    st.markdown("### ⚡ Quick Actions")
    
    # Context-aware quick actions
    if section == "Chat":
        if st.button("🎲 Random Question", use_container_width=True, key="sidebar_random"):
            if 'suggested_question' not in st.session_state:
                questions = [
                    "What are the main concepts?",
                    "Explain the key topics",
                    "What should I focus on?",
                    "How do concepts connect?"
                ]
                import random
                st.session_state.suggested_question = random.choice(questions)
                st.rerun()
    
    elif section == "Quiz":
        if st.button("🎯 Quick Quiz (5Q)", use_container_width=True, key="sidebar_quick_quiz"):
            st.session_state.quick_quiz_requested = True
            st.rerun()
    
    elif section == "Summary":
        if st.button("📊 Full Summary", use_container_width=True, key="sidebar_full_summary"):
            st.session_state.full_summary_requested = True
            st.rerun()
    
    # Universal quick actions
    if st.button("🔄 Refresh Page", use_container_width=True, key="refresh_page"):
        st.rerun()
    
    st.markdown("---")
    
    # Help & Tips Section
    with st.expander("💡 Tips & Help", expanded=False):
        st.markdown("""
        **Getting Started:**
        1. 📤 Upload your course files in Setup
        2. 💬 Chat to ask questions
        3. 📝 Generate quizzes to test yourself
        4. 📋 Create summaries for review
        
        **Pro Tips:**
        - Use specific topic names for better results
        - Ask follow-up questions for deeper understanding
        - Export your quizzes and summaries for offline study
        - 🚀 Cache speeds up repeated operations
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
        <p>🤖 Powered by AI<br>
        📚 Your study companion<br>
        ⚡ Cache-optimized</p>
    </div>
    """, unsafe_allow_html=True)

# Updated Admin section with better UX and cache clearing
if section == "Admin (Upload/Index)":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h3 style="margin: 0;">📤 Upload Course Materials</h3>
        <p style="margin: 5px 0 0 0;">Upload your PDFs and text files to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show current course status
    if course and course != "Demo Course":
        status = check_course_status(course)
        if status["is_ready"]:
            st.info(f"☁️ Current course has {status['document_count']} documents in ChromaDB Cloud")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Course Data", type="secondary"):
                    if delete_course(course):
                        st.success("Course data cleared from cloud!")
                        st.rerun()
                    else:
                        st.error("Failed to clear course data")
            
            with col2:
                if st.button("📊 View All Courses", type="secondary"):
                    courses = list_all_courses()
                    if courses:
                        st.write("**Available courses:**")
                        for c in courses:
                            st.write(f"• {c}")
                    else:
                        st.write("No courses found")
    
    # File upload with better guidance
    st.markdown("### 📁 Select Your Files")
    ups = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF and TXT files. You can upload multiple files at once."
    )
    
    # Show file preview
    if ups:
        st.markdown("### 📋 Files to Process")
        total_size = 0
        for i, file in enumerate(ups, 1):
            file_size = len(file.getvalue()) / 1024  # KB
            total_size += file_size
            st.info(f"📄 **{i}.** {file.name} ({file_size:.1f} KB)")
        
        st.caption(f"Total size: {total_size:.1f} KB")
        
        # Process button with better styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("⚙️ Processing files...")
                    progress_bar.progress(20)
                    
                    status_text.text("🔍 Extracting text...")
                    progress_bar.progress(40)
                    
                    status_text.text("🤖 Creating embeddings...")
                    progress_bar.progress(60)
                    
                    added = save_upload_and_index(course, ups)
                    progress_bar.progress(80)
                    
                    status_text.text("✅ Finalizing...")
                    progress_bar.progress(100)
                    
                    if added > 0:
                        st.success(f"🎉 Successfully indexed {added} text chunks!")
                        st.balloons()
                        st.info("✅ You can now use Chat, Quiz, Summary, and Interview modes.")
                        
                        # Show updated status
                        status = check_course_status(course)
                        st.metric("📚 Documents Indexed", status['document_count'])
                    else:
                        st.error("❌ No content was indexed. Please check your files.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing files: {e}")
                finally:
                    progress_bar.empty()
                    status_text.empty()
    
    else:
        # Upload guidance
        st.markdown("""
        <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 30px; text-align: center; margin: 20px 0;">
            <h4 style="color: #666;">📁 Drag & Drop Files Here</h4>
            <p style="color: #888;">Or click "Browse files" above to select your course materials</p>
            <p style="color: #aaa; font-size: 12px;">Supported: PDF, TXT • Multiple files allowed</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Examples of good files to upload
        st.markdown("### 💡 What to Upload")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **✅ Good files:**
            - 📖 Lecture notes
            - 📚 Textbook chapters  
            - 📝 Study guides
            - 🔬 Lab manuals
            """)
        
        with col2:
            st.markdown("""
            **💡 Tips:**
            - Clear, readable text works best
            - Multiple small files > one huge file
            - PDF text should be selectable
            - Organize by topic if possible
            """)

elif section == "Chat":
    st.subheader("💬 Chat with your course notes")
    
    init_chat_state()
    
    # Check if course is ready - no more dimension issues!
    status = check_course_status(course)
    if "error" in status:
        st.error(f"❌ Database error: {status['error']}")
        st.info("Please check your ChromaDB Cloud connection and try again.")
        st.stop()
    elif not status["is_ready"]:
        st.warning("⚠️ No course materials found. Please upload files in the Setup section first.")
        st.stop()
    
    # Simple controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_messages = []
            st.session_state.last_sources = []
            st.rerun()
    
    with col2:
        show_sources = st.checkbox("📚 Show Sources", value=True)
    
    with col3:
        st.caption(f"⚡ {status['document_count']} docs ready")
    
    # Chat history
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about this course...")
    
    # Handle sidebar quick actions
    if hasattr(st.session_state, 'suggested_question') and st.session_state.suggested_question:
        user_input = st.session_state.suggested_question
        del st.session_state.suggested_question
    
    if user_input:
        # Add user message
        add_chat_message("user", user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get context with performance timing
        start_time = st.empty()
        with st.spinner("🔍 Searching notes..."):
            import time
            start = time.time()
            context_text, docs = build_context_block(course, user_input, k=5)
            search_time = time.time() - start
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed_text = ""
            
            if not context_text.strip():
                streamed_text = "I don't find this in the class notes."
                placeholder.markdown(streamed_text)
                add_chat_message("assistant", streamed_text)
                st.session_state.last_sources = []
            else:
                # Build system message with context
                sys_msg = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_text}"
                sys = SystemMessage(content=sys_msg)
                
                # Build conversation history (last 8 messages)
                history_msgs = []
                for m in st.session_state.chat_messages[-8:]:
                    if m["role"] == "user":
                        history_msgs.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        history_msgs.append(AIMessage(content=m["content"]))
                
                lc_messages = [sys, *history_msgs, HumanMessage(content=user_input)]
                llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), temperature=0.3)
                
                try:
                    # Stream response
                    for chunk in llm.stream(lc_messages):
                        chunk_content = chunk.content or ""
                        streamed_text += chunk_content
                        placeholder.markdown(streamed_text + "▌")
                    
                    placeholder.markdown(streamed_text)
                    
                except Exception as e:
                    # Fallback to non-streaming
                    try:
                        resp = llm.invoke(lc_messages)
                        streamed_text = (resp.content or "").strip()
                        placeholder.markdown(streamed_text)
                    except:
                        streamed_text = "Sorry, I'm having trouble connecting. Please try again."
                        placeholder.markdown(streamed_text)
                
                add_chat_message("assistant", streamed_text)
                st.session_state.last_sources = [d.metadata.get("source", "Unknown") for d in docs]
        
        # Show sources if enabled
        if show_sources and st.session_state.last_sources:
            with st.expander(f"📚 Sources ({len(st.session_state.last_sources)} documents) - Search took {search_time:.2f}s"):
                for source in st.session_state.last_sources:
                    st.caption(f"• {source}")
        
        # Performance info for debugging
        if st.checkbox("🔧 Show Performance Info", key="perf_debug"):
            st.caption(f"Search time: {search_time:.2f}s | Context length: {len(context_text)} chars")
    
    # Welcome message for new users
    if not st.session_state.chat_messages:
        st.info("👋 Welcome! Ask me anything about your course materials. I'll search through your uploaded notes to help answer your questions.")
        
        # Simple starter questions
        st.markdown("**💡 Try asking:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📖 What are the main topics?", use_container_width=True):
                st.session_state.suggested_question = "What are the main topics in my notes?"
                st.rerun()
            
            if st.button("🎯 Key concepts to remember?", use_container_width=True):
                st.session_state.suggested_question = "What are the key concepts I should remember?"
                st.rerun()
        
        with col2:
            if st.button("📝 Create a study plan", use_container_width=True):
                st.session_state.suggested_question = "Can you create a study plan for me?"
                st.rerun()
            
            if st.button("🤔 What might be challenging?", use_container_width=True):
                st.session_state.suggested_question = "What topics might be challenging to understand?"
                st.rerun()

elif section == "Quiz":
    st.subheader("📝 Generate a Quiz from Your Notes")
    
    # Check if course is ready
    status = check_course_status(course)
    if not status["is_ready"]:
        st.warning("⚠️ No course materials found. Please upload files in the Setup section first.")
        st.stop()
    
    # Quiz configuration
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("Topic / focus", value="Core concepts", 
                             help="Specify a topic or leave as 'Core concepts' for general questions")
    with col2:
        num_q = st.slider("Number of questions", 3, 15, 5)
    
    # Difficulty selection
    difficulty = st.selectbox("Difficulty Level", 
                             ["Mixed", "Easy", "Medium", "Hard"],
                             help="Choose question difficulty or mix of all levels")
    
    # Initialize session state for quiz
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    
    # Handle quick quiz request
    if hasattr(st.session_state, 'quick_quiz_requested') and st.session_state.quick_quiz_requested:
        del st.session_state.quick_quiz_requested
        topic = "Core concepts"
        num_q = 5
    
    # Generate quiz button
    if st.button("🎯 Generate Quiz", type="primary") or hasattr(st.session_state, 'quick_quiz_requested'):
        with st.spinner("🔍 Retrieving notes and generating quiz..."):
            ctx, *_ = build_context_block(course, topic)
            
            if not ctx.strip():
                st.warning("⚠️ No relevant notes found. Upload files first or try a different topic.")
            else:
                # Generate quiz with retry logic
                with st.spinner("🤖 Creating questions..."):
                    qs = quiz_from_context(context=ctx, count=num_q, topic=topic)
                    
                if not qs or len(qs) == 0:
                    st.error("❌ Quiz generation failed. Please try again with a different topic.")
                else:
                    st.session_state.quiz_data = qs
                    st.session_state.quiz_submitted = False
                    st.session_state.user_answers = {}
                    st.success(f"✅ Generated {len(qs)} questions successfully!")
                    st.rerun()
    
    # Display quiz if generated
    if st.session_state.quiz_data:
        st.markdown("---")
        st.subheader("📋 Quiz Questions")
        
        qs = st.session_state.quiz_data
        
        # Quiz form
        with st.form("quiz_form"):
            user_answers = {}
            
            for i, q in enumerate(qs, 1):
                st.markdown(f"### Question {i}")
                question_text = q.get('question', f'Question {i}')
                st.write(f"**{question_text}**")
                
                # Handle options with better formatting
                options = q.get("options", [f"A) Option {j}" for j in range(1, 5)])
                labels = ["A", "B", "C", "D"]
                clean_options = []
                
                # Clean option text
                for j, opt in enumerate(options):
                    if isinstance(opt, str):
                        # Remove A), B), etc. prefixes if present
                        if opt.strip().startswith(f"{labels[j]})"):
                            clean_text = opt.split(")", 1)[1].strip()
                        elif ")" in opt[:3]:
                            clean_text = opt.split(")", 1)[1].strip()
                        else:
                            clean_text = opt.strip()
                    else:
                        clean_text = str(opt)
                    
                    clean_options.append(f"{labels[j]}) {clean_text}")
                
                # Display options as radio buttons
                choice = st.radio(
                    "Select your answer:",
                    options=labels,
                    format_func=lambda x: clean_options[labels.index(x)],
                    key=f"q{i}",
                    horizontal=True
                )
                user_answers[i] = choice
                st.markdown("---")
            
            # Submit quiz button
            submitted = st.form_submit_button("📊 Submit Quiz", type="primary")
            
            if submitted:
                st.session_state.user_answers = user_answers
                st.session_state.quiz_submitted = True
        
        # Show results if submitted
        if st.session_state.quiz_submitted and st.session_state.user_answers:
            st.markdown("---")
            st.subheader("🎉 Quiz Results")
            
            score = 0
            total_questions = len(qs)
            detailed_results = []
            
            for i, q in enumerate(qs, 1):
                user_choice = st.session_state.user_answers.get(i, "")
                correct_answer = q.get("answer", "A").upper()
                is_correct = user_choice.upper() == correct_answer
                
                if is_correct:
                    score += 1
                
                detailed_results.append({
                    'question_num': i,
                    'question': q.get('question', ''),
                    'user_answer': user_choice,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'options': q.get('options', [])
                })
            
            # Score display
            percentage = (score / total_questions) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", f"{score}/{total_questions}")
            with col2:
                st.metric("Percentage", f"{percentage:.1f}%")
            with col3:
                if percentage >= 80:
                    st.success("Excellent! 🌟")
                elif percentage >= 60:
                    st.info("Good job! 👍")
                else:
                    st.warning("Keep studying! 📚")
            
            # Detailed feedback
            st.subheader("📈 Detailed Results")
            
            for result in detailed_results:
                i = result['question_num']
                
                if result['is_correct']:
                    st.success(f"✅ **Question {i}**: Correct!")
                else:
                    st.error(f"❌ **Question {i}**: Incorrect")
                
                with st.expander(f"Review Question {i}", expanded=not result['is_correct']):
                    st.write(f"**Question:** {result['question']}")
                    
                    # Show options with indicators
                    options = result['options']
                    labels = ["A", "B", "C", "D"]
                    
                    for j, opt in enumerate(options):
                        label = labels[j]
                        # Clean option text
                        if isinstance(opt, str) and ")" in opt[:3]:
                            clean_text = opt.split(")", 1)[1].strip()
                        else:
                            clean_text = str(opt)
                        
                        if label == result['correct_answer']:
                            st.success(f"✅ {label}) {clean_text} **(Correct Answer)**")
                        elif label == result['user_answer']:
                            st.error(f"❌ {label}) {clean_text} **(Your Answer)**")
                        else:
                            st.write(f"◯ {label}) {clean_text}")
                
                st.markdown("---")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Retake Quiz"):
                    st.session_state.quiz_submitted = False
                    st.session_state.user_answers = {}
                    st.rerun()
            
            with col2:
                if st.button("📝 Generate New Quiz"):
                    st.session_state.quiz_data = None
                    st.session_state.quiz_submitted = False
                    st.session_state.user_answers = {}
                    st.rerun()
            
            with col3:
                # Export results button
                if st.button("📤 Export Results"):
                    results_text = f"Quiz Results - {topic}\n"
                    results_text += f"Score: {score}/{total_questions} ({percentage:.1f}%)\n\n"
                    
                    for result in detailed_results:
                        results_text += f"Q{result['question_num']}: {result['question']}\n"
                        results_text += f"Your answer: {result['user_answer']}\n"
                        results_text += f"Correct answer: {result['correct_answer']}\n"
                        results_text += f"Result: {'Correct' if result['is_correct'] else 'Incorrect'}\n\n"
                    
                    st.download_button(
                        label="Download Results",
                        data=results_text,
                        file_name=f"quiz_results_{topic.replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
    
    # Help section
    if not st.session_state.quiz_data:
        st.markdown("---")
        st.info("""
        💡 **How to use:**
        1. Enter a specific topic or keep "Core concepts" for general questions
        2. Choose the number of questions (3-15)
        3. Select difficulty level
        4. Click "Generate Quiz" to create questions from your uploaded notes
        5. Answer all questions and click "Submit Quiz" to see your results
        """)
        
        # Show sample of available topics if context exists
        if st.button("🔍 Show Available Topics"):
            try:
                ctx, *_ = build_context_block(course, "")  # Get all content
                if ctx.strip():
                    # Extract some key terms as suggested topics
                    import re
                    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', ctx)
                    common_topics = list(set(words))[:10]
                    if common_topics:
                        st.write("**Suggested topics based on your notes:**")
                        st.write(", ".join(common_topics))
            except:
                st.write("Upload some notes to see suggested topics.")

elif section == "Summary":
    st.subheader("📋 Generate concise summary notes")
    
    # Check if course is ready
    status = check_course_status(course)
    if not status["is_ready"]:
        st.warning("⚠️ No course materials found. Please upload files in the Setup section first.")
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("Topic / section", value="Overview")
    with col2:
        st.caption(f"⚡ {status['document_count']} docs available")
    
    # Handle full summary request
    if hasattr(st.session_state, 'full_summary_requested') and st.session_state.full_summary_requested:
        del st.session_state.full_summary_requested
        topic = "Overview"
    
    if st.button("📊 Generate Summary", type="primary") or hasattr(st.session_state, 'full_summary_requested'):
        with st.spinner("🔍 Retrieving notes…"):
            ctx, _ = build_context_block(course, topic)
        if not ctx.strip():
            st.warning("⚠️ No relevant notes found. Upload files first or try a different topic.")
        else:
            with st.spinner("📝 Creating summary..."):
                s = summary_from_context(ctx, topic=topic)
            
            st.markdown("---")
            st.markdown(s)
            
            # Download button
            st.download_button(
                label="📥 Download Summary",
                data=s,
                file_name=f"summary_{topic.replace(' ', '_')}.md",
                mime="text/markdown"
            )

elif section == "Interview Qs":
    st.subheader("🎤 Generate interview/exam questions with ideal answers")
    
    # Check if course is ready
    status = check_course_status(course)
    if not status["is_ready"]:
        st.warning("⚠️ No course materials found. Please upload files in the Setup section first.")
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("Topic / area", value="Important topics")
    with col2:
        count = st.slider("How many?", 5, 20, 10)
    
    if st.button("🎯 Generate Q&A", type="primary"):
        with st.spinner("🔍 Retrieving notes…"):
            ctx, _ = build_context_block(course, topic)
        if not ctx.strip():
            st.warning("⚠️ No relevant notes found. Upload files first or try a different topic.")
        else:
            with st.spinner("🤖 Creating interview questions..."):
                qa = interview_qs_from_context(ctx, topic, count=count)
            
            st.markdown("---")
            st.markdown(qa)
            
            # Download button
            st.download_button(
                label="📥 Download Q&A",
                data=qa,
                file_name=f"interview_qa_{topic.replace(' ', '_')}.md",
                mime="text/markdown"
            )