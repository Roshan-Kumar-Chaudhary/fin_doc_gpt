# qa/hybrid_qa.py
from transformers import pipeline
from typing import Union

class HybridQASystem:
    def __init__(self):
        self.qa_model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )
        self.financial_terms = {
            'gross income': ['gross income', 'total revenue'],
            'net profit': ['net income', 'net earnings']
        }

    def answer_question(self, question: str, context: str) -> Dict:
        """Enhanced Q&A with financial term handling"""
        # Check if question contains financial terms
        for term, variants in self.financial_terms.items():
            if any(v in question.lower() for v in variants):
                return self._answer_financial_question(term, context)
        
        # Fallback to standard Q&A
        return self.qa_model(question=question, context=context)

    def _answer_financial_question(self, term: str, context: str) -> Dict:
        """Special handling for financial metrics"""
        # Custom logic to extract financial values
        pass