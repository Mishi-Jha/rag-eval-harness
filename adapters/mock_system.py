class MockRetriever:
    def retrieve(self,question:str)->list[str]:
        return["Mock chunk 1","Mock chunk 2", "Mock chunk 3"]
class MockGenerator:
    def __init__(self):
        self.last_usage={"input_tokens":100,"output_tokens":50}
    def generate(self,question:str,chunks:list[str])->str:
        return "This is a mock answer"   