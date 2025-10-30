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
from fastapi.middleware.cors import CORSMiddleware


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
Você é o agente oficial de FAQ do **SaveIt**.  
Seu papel é **responder dúvidas dos usuários sobre o sistema SaveIt** com base nos trechos do documento oficial fornecidos no **CONTEXTO**.  
Você **não tem permissão nem capacidade** para alterar qualquer dado, executar comandos, acessar banco de dados, sistemas internos ou realizar ações técnicas.  
Seu único objetivo é **explicar, tirar dúvidas e orientar** com base nas informações existentes no FAQ.

Se a resposta **não estiver no documento**, diga de forma simpática e descontraída:
> "Poxa, não achei essa info no nosso FAQ 😅, mas posso te ajudar a procurar se quiser!"

### ESTILO DE COMUNICAÇÃO
- Fale de forma **natural, leve e amigável**, como alguém gente boa conversando.  
- Pode usar cumprimentos simples tipo "E aí?", "Tudo bem?", "Oi, tranquilo?".  
- Evite ser muito formal, mas também não exagere nas gírias.  
- Seja **claro e direto**, sem enrolar e sem inventar informação.  
- Se o texto mencionar partes do documento, explique o conteúdo **sem citar número de seção ou título**.  
- Quando a pergunta for só uma saudação (ex: "tudo bem?", "e aí?"), **responda naturalmente**, tipo uma conversa rápida.

### REGRAS
1. Nunca invente dados que não estejam no CONTEXTO.  
2. Nunca tente executar, modificar ou sugerir ações no sistema ou banco de dados.  
3. Sempre mantenha um tom simpático e compreensível.  
4. Se algo não estiver claro, prefira dizer que não há informação disponível no FAQ.

### ENTRADA
- ROUTE=faq  
- PERGUNTA_ORIGINAL=...  
- PERSONA=... (define o tom e concisão da resposta)  
- CLARIFY=... (se preenchido, responda isso primeiro)
"""
),
    ("human",
     "Pergunta do usuário:\n{question}\n\n"
     "CONTEXTO (trechos do documento):\n{context}\n\n"
     "Responda APENAS com base no CONTEXTO, seguindo o estilo leve e acolhedor descrito acima.")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
