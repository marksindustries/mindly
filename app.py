import os
import streamlit as st
from dotenv import load_dotenv

from rag_chain import save_upload_and_index, get_retriever
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


def init_chat_state():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

def add_chat_message(role, content):
    st.session_state.chat_messages.append({"role": role, "content": content})

def build_context_block(course_name: str, user_query: str, k: int = 5):
    """Retrieve top-k chunks for the query and format a short context block."""
    retriever = get_retriever(course_name)
    docs = retriever.get_relevant_documents(user_query)
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


# ---------------- Sidebar ----------------
# st.sidebar.header("Course")
# course = st.sidebar.text_input("Course name", value="Demo Course", help="e.g., 'JEE Physics 2025', 'Data Structures'")
# st.sidebar.markdown("---")
# section = st.sidebar.radio("Mode", ["Admin (Upload/Index)", "Chat", "Quiz", "Summary", "Interview Qs"])

# # ---------------- Sections ----------------
# if section == "Admin (Upload/Index)":
#     st.subheader("Upload curriculum files (PDF/TXT)")
#     ups = st.file_uploader("Upload files", type=["pdf","txt"], accept_multiple_files=True)
#     if ups and st.button("Index Files", type="primary"):
#         with st.spinner("Indexing..."):
#             added = save_upload_and_index(course, ups)
#         st.success(f"Indexed {added} text chunks. You can now use Chat/Quiz/Summary/Interview tabs.")
# Enhanced Sidebar with better UI/UX
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
    
    # Visual indicator of course status
    if course and course != "Demo Course":
        st.success(f"✅ Working with: **{course}**")
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
            files_indexed = st.button("📊", help="Check indexed files", key="check_files")
    
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
    
    # Course material indicator
    try:
        # Try to check if there's indexed content (simplified check)
        retriever = get_retriever(course)
        test_docs = retriever.get_relevant_documents("test")
        if test_docs:
            st.success("📚 Materials Ready")
        else:
            st.warning("📤 Upload Materials First")
    except:
        st.info("📤 No Materials Yet")
    
    st.markdown("---")
    
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
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
        <p>🤖 Powered by AI<br>
        📚 Your study companion</p>
    </div>
    """, unsafe_allow_html=True)

# Updated Admin section with better UX
if section == "Admin (Upload/Index)":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h3 style="margin: 0;">📤 Upload Course Materials</h3>
        <p style="margin: 5px 0 0 0;">Upload your PDFs and text files to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        for i, file in enumerate(ups, 1):
            file_size = len(file.getvalue()) / 1024  # KB
            st.info(f"📄 **{i}.** {file.name} ({file_size:.1f} KB)")
        
        # Process button with better styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                with st.spinner("⚙️ Processing and indexing files..."):
                    added = save_upload_and_index(course, ups)
                
                if added > 0:
                    st.success(f"🎉 Successfully indexed {added} text chunks!")
                    st.balloons()
                    st.info("✅ You can now use Chat, Quiz, Summary, and Interview modes.")
                else:
                    st.error("❌ No content was indexed. Please check your files.")
    
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
    
    # Simple controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_messages = []
            st.session_state.last_sources = []
            st.rerun()
    
    with col2:
        show_sources = st.checkbox("📚 Show Sources", value=True)
    
    # Chat history
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about this course...")
    
    if user_input:
        # Add user message
        add_chat_message("user", user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get context and generate response
        with st.spinner("🔍 Searching notes..."):
            context_text, docs = build_context_block(course, user_input, k=5)
        
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
            with st.expander(f"📚 Sources ({len(st.session_state.last_sources)} documents)"):
                for source in st.session_state.last_sources:
                    st.caption(f"• {source}")
    
    # Welcome message for new users
    if not st.session_state.chat_messages:
        st.info("👋 Welcome! Ask me anything about your course materials. I'll search through your uploaded notes to help answer your questions.")
        
        # Simple starter questions
        st.markdown("**💡 Try asking:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📖 What are the main topics?", use_container_width=True):
                add_chat_message("user", "What are the main topics in my notes?")
                st.rerun()
            
            if st.button("🎯 Key concepts to remember?", use_container_width=True):
                add_chat_message("user", "What are the key concepts I should remember?")
                st.rerun()
        
        with col2:
            if st.button("📝 Create a study plan", use_container_width=True):
                add_chat_message("user", "Can you create a study plan for me?")
                st.rerun()
            
            if st.button("🤔 What might be challenging?", use_container_width=True):
                add_chat_message("user", "What topics might be challenging to understand?")
                st.rerun()

elif section == "Quiz":
    st.subheader("📝 Generate a Quiz from Your Notes")
    
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
    
    # Generate quiz button
    if st.button("🎯 Generate Quiz", type="primary"):
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
    st.subheader("Generate concise summary notes")
    topic = st.text_input("Topic / section", value="Overview")
    if st.button("Generate Summary"):
        with st.spinner("Retrieving notes…"):
            ctx, _ = build_context_block(course, topic)
        if not ctx.strip():
            st.warning("No relevant notes found. Upload files first.")
        else:
            s = summary_from_context(ctx, topic=topic)
            st.markdown(s)

elif section == "Interview Qs":
    st.subheader("Generate interview/exam questions with ideal answers")
    topic = st.text_input("Topic / area", value="Important topics")
    count = st.slider("How many?", 5, 20, 10)
    if st.button("Generate Q&A"):
        with st.spinner("Retrieving notes…"):
            ctx, _ = build_context_block(course, topic)
        if not ctx.strip():
            st.warning("No relevant notes found. Upload files first.")
        else:
            qa = interview_qs_from_context(ctx,topic,count=count)
            st.markdown(qa)
