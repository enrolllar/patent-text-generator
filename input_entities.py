import json
from io import StringIO
from pydantic import BaseModel
class Token(BaseModel):
    text: str
    start: int
    end: int
    token_start: int
    token_end: int
    entityLabel: str
    propertiesList: list[str]
    commentsList: list[str]

class Relation(BaseModel):
    child: int
    head: int
    relationLabel: str
    propertiesList: list[str]
    commentsList: list[str]

class Document(BaseModel):
    documentName: str
    document: str
    tokens: list[Token]
    relations: list[Relation]
    token_map: dict[int, Token] = dict()


class DocumentsList(BaseModel):
    items: list[Document]
    @staticmethod
    def read_documents_from_file(file_path: str):
        file = open(file_path, encoding='utf-8')
        text = file.read()
        file.close()
        documents = DocumentsList(items=json.load(StringIO(text))).items

        for doc in documents:
            for token in doc.tokens:
                doc.token_map[token.token_start] = token

        return documents
