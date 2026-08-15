from contracts.base import Retriever, Generator
class Harness:
    def __init__(self,retriever:Retriever,generator:Generator):
        self.retriever=retriever
        self.generator=generator
    def run(self, question:str)->str:
        chunks=self.retriever.retrieve(question)
        result=self.generator.generate(question,chunks)
        return result