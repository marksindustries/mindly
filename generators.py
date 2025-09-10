import os, json, re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from prompts import (
    RAG_ANSWER_PROMPT,
    SUMMARY_PROMPT,
    INTERVIEW_QS_PROMPT,
    QUIZ_JSON_PROMPT,
)

load_dotenv()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

def _chat(temp: float = 0.2) -> ChatGroq:
    # ChatGroq reads GROQ_API_KEY from env
    return ChatGroq(model=GROQ_MODEL, temperature=temp)

def rag_answer(question: str, context: str) -> str:
    chain = RAG_ANSWER_PROMPT | _chat(0.2) | StrOutputParser()
    return chain.invoke({"context": context, "question": question}).strip()

def summary_from_context(context: str, topic:str) -> str:
    chain = SUMMARY_PROMPT | _chat(0.2) | StrOutputParser()
    return chain.invoke({"context": context, "topic":topic}).strip()

def interview_qs_from_context(context: str,topic:str ,count: int = 10) -> str:
    chain = INTERVIEW_QS_PROMPT | _chat(0.2) | StrOutputParser()
    return chain.invoke({"context": context, "count": count, "topic":topic}).strip()

def quiz_from_context(context: str,topic: str ,count: int = 5, max_retries: int = 3):
    """
    Generate quiz questions with improved error handling and JSON parsing
    """
    for attempt in range(max_retries):
        try:
            # Use higher temperature for more creativity in options
            chain = QUIZ_JSON_PROMPT | _chat(0.3) | StrOutputParser()
            raw = chain.invoke({"context": context, "count": count,"topic":topic}).strip()
            
            # Clean the response - remove any markdown formatting
            raw = raw.replace("```json", "").replace("```", "").strip()
            
            # Try to parse as JSON first
            try:
                data = json.loads(raw)
                if isinstance(data, list) and len(data) > 0:
                    # Validate that each question has proper structure
                    valid_questions = []
                    for item in data[:count]:
                        if (isinstance(item, dict) and 
                            'question' in item and 
                            'options' in item and 
                            'answer' in item and
                            isinstance(item['options'], list) and
                            len(item['options']) == 4):
                            valid_questions.append(item)
                    
                    if valid_questions:
                        return valid_questions
                        
            except json.JSONDecodeError:
                pass
            
            # If JSON parsing fails, try structured text parsing
            parsed_questions = parse_structured_quiz_text(raw, count)
            if parsed_questions:
                return parsed_questions
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Return fallback questions if all attempts fail
                return generate_fallback_quiz(context, count)
    
    return generate_fallback_quiz(context, count)

def parse_structured_quiz_text(text: str, count: int) -> list:
    """
    Parse quiz text when JSON parsing fails
    """
    questions = []
    
    # Split by question patterns
    question_blocks = re.split(r'\n*(?:Question\s*\d+|Q\d+|^\d+\.)', text, flags=re.MULTILINE)
    
    for block in question_blocks[1:]:  # Skip first empty block
        if not block.strip():
            continue
            
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 5:  # Need at least question + 4 options
            continue
            
        question_text = lines[0].rstrip(':?')
        options = []
        answer = "A"
        
        # Extract options (looking for A), B), C), D) pattern)
        option_pattern = re.compile(r'^[A-D]\)\s*(.+)', re.IGNORECASE)
        for line in lines[1:]:
            match = option_pattern.match(line)
            if match:
                options.append(line)
                if len(options) == 4:
                    break
        
        # Look for answer
        for line in lines:
            answer_match = re.search(r'answer[:\s]*([A-D])', line, re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).upper()
                break
        
        if len(options) == 4:
            questions.append({
                "question": question_text,
                "options": options,
                "answer": answer
            })
            
        if len(questions) >= count:
            break
    
    return questions[:count]

def generate_fallback_quiz(context: str, count: int) -> list:
    """
    Generate basic quiz questions when parsing fails
    """
    # Use a simpler prompt that's more likely to work
    fallback_prompt = f"""
    Create {count} multiple choice questions based on this context. 
    Use this EXACT format for each question:
    
    Q: [question text]
    A) [option A]
    B) [option B] 
    C) [option C]
    D) [option D]
    Answer: [A/B/C/D]
    
    Context: {context[:1000]}...
    """
    
    try:
        chain = ChatGroq(model=GROQ_MODEL, temperature=0.4) | StrOutputParser()
        result = chain.invoke(fallback_prompt)
        return parse_structured_quiz_text(result, count)
    except:
        # Ultimate fallback - return template questions
        return [{
            "question": f"Sample question {i+1} based on the provided context",
            "options": [
                f"A) Option A for question {i+1}",
                f"B) Option B for question {i+1}",
                f"C) Option C for question {i+1}", 
                f"D) Option D for question {i+1}"
            ],
            "answer": "A"
        } for i in range(count)]

# Alternative function using JsonOutputParser (might work better with some models)
def quiz_from_context_json_parser(context: str, count: int = 5):
    """
    Alternative implementation using LangChain's JsonOutputParser
    """
    try:
        parser = JsonOutputParser()
        chain = QUIZ_JSON_PROMPT | _chat(0.3) | parser
        result = chain.invoke({"context": context, "count": count})
        
        if isinstance(result, list):
            return result[:count]
        
    except Exception as e:
        print(f"JsonOutputParser failed: {e}")
        return quiz_from_context(context, count)

# Test function to validate quiz output
def test_quiz_output(quiz_data):
    """
    Test function to validate quiz structure
    """
    if not isinstance(quiz_data, list):
        return False, "Quiz data is not a list"
    
    for i, q in enumerate(quiz_data):
        if not isinstance(q, dict):
            return False, f"Question {i+1} is not a dictionary"
        
        required_keys = ['question', 'options', 'answer']
        for key in required_keys:
            if key not in q:
                return False, f"Question {i+1} missing key: {key}"
        
        if not isinstance(q['options'], list) or len(q['options']) != 4:
            return False, f"Question {i+1} doesn't have exactly 4 options"
        
        if q['answer'] not in ['A', 'B', 'C', 'D']:
            return False, f"Question {i+1} has invalid answer: {q['answer']}"
    
    return True, "All questions valid"

