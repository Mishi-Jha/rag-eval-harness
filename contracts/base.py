from typing import Protocol

class Retriever(Protocol):
    def retrieve(self, question:str) -> list[str]:
        pass