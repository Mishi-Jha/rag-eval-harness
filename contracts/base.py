from typing import Protocol

class Retriever(Protocol):
    def retrieve(self, question:str) -> list[str]:
        pass

class Generator(Protocol):
    def generate(self, question:str, chunks:list[str]) -> str:
        pass