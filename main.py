import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from flask import Flask, request, jsonify
from faq_tool import get_faq_context

load_dotenv(dotenv_path=".env")

llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

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

### ENTRADAb n
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

faq_chain_core = (
    RunnablePassthrough.assign(
        question=itemgetter("input"),
        context=lambda x: get_faq_context(x["input"])
    )
    | prompt_faq
    | llm_fast
    | StrOutputParser()
)

app = Flask(__name__)

def answer_question(question, session_id="default"):
    return faq_chain_core.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )

@app.route("/faq", methods=["POST"])
def faq_endpoint():
    payload = request.get_json(force=True)
    question = payload.get("question") or payload.get("input")
    if not question:
        return jsonify({"error": "Campo 'question' é obrigatório."}), 400
    try:
        resp = answer_question(question, session_id=payload.get("session_id", "WEB"))
        return jsonify({"answer": resp})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Rodar servidor HTTP")
    args = parser.parse_args()
    if args.serve:
        app.run(host="0.0.0.0", port=5000)
    else:
        print("FAQ assistant (digite 'sair' para encerrar)"),
        while True:
            q = input("Pergunta: ").strip()
            if q.lower() in ("sair", "exit", "quit"):
                break
            try:
                print(answer_question(q))
            except Exception as e:
                print("Erro:", e)