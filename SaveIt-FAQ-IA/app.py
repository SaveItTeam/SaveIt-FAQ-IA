import os
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from faq_tool import get_faq_context

# Carrega variáveis de ambiente
load_dotenv(dotenv_path=".env")

# 1. Configuração do Modelo de Linguagem (LLM)
llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# 2. Definição do Prompt do Sistema
system_prompt_faq = ChatPromptTemplate.from_messages([
    ("system",
     """
### PAPEL
Você deve responder perguntas sobre dúvidas SOMENTE com base no documento normativo oficial (trechos fornecidos em CONTEXTO).
Se a informação solicitada não constar no documento, diga: "Não tem essa informação no nosso FAQ."

### REGRAS 
- Seja breve, claro e educado.
- Fale em linguagem simples, sem jargões técnicos.
- Se o trecho citar seção, mencione a parte relevante sem ser o número da seção.
- Não invente informações.

### ENTRADA
- ROUTE=faq
- PERGUNTA_ORIGINAL=...
- PERSONA=... (diretriz de concisão)
- CLARIFY=... (se preenchido, responda primeiro)
"""
),
    ("human",
     "Pergunta do usuário:\n{question}\n\nCONTEXTO (trechos do documento):\n{context}\n\nResponda com base APENAS no CONTEXTO.")
])

prompt_faq = ChatPromptTemplate.from_messages([
    system_prompt_faq,
    ("human",
     "Pergunta do usuário:\n{question}\n\n"
     "CONTEXTO (trechos do documento):\n{context}\n\n"
     "Responda com base APENAS no CONTEXTO."
     )
])

# 3. Cadeia RAG (Retrieval-Augmented Generation)
faq_chain_core = (
    RunnablePassthrough.assign(
        question=itemgetter("input"),
        context=lambda x: get_faq_context(x["input"])
    )
    | prompt_faq
    | llm_fast
    | StrOutputParser()
)

def answer_question(question, session_id="default"):
    return faq_chain_core.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )

# 4. Inicialização do FastAPI
app = FastAPI(
    title="SaveIt FAQ AI API",
    description="API para responder perguntas sobre o SaveIt usando RAG (Retrieval-Augmented Generation) e Google Gemini.",
    version="1.0.0"
)

# 5. Definição do Schema de Requisição
class QuestionRequest(BaseModel):
    question: str
    session_id: str = "WEB" # Opcional, mantido para compatibilidade

# 6. Definição do Endpoint
@app.post("/faq")
async def faq_endpoint(request_data: QuestionRequest):
    if not request_data.question:
        raise HTTPException(status_code=400, detail="Campo 'question' é obrigatório.")
    try:
        # A função answer_question é síncrona, mas o FastAPI a gerencia em um thread pool
        resp = answer_question(request_data.question, session_id=request_data.session_id)
        return {"answer": resp}
    except Exception as e:
        # Em caso de erro (ex: chave da API inválida, problema no RAG), retorna 500
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {str(e)}")
