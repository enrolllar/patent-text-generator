#=============================================== 3 модели + визуализация графов + выбор в начале ==================================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations
from pydantic import BaseModel
from typing import List, Dict
import json
import networkx as nx
import requests
from collections import defaultdict
from g4f.client import Client
import graphviz
import os
import nest_asyncio

nest_asyncio.apply()

# === МОДЕЛИ ДАННЫХ ===
class Token(BaseModel):
    text: str
    start: int
    end: int
    token_start: int
    token_end: int
    entityLabel: str
    propertiesList: List[str] = []
    commentsList: List[str] = []

class Relation(BaseModel):
    child: int
    head: int
    relationLabel: str
    propertiesList: List[str] = []
    commentsList: List[str] = []

class Document(BaseModel):
    documentName: str
    document: str
    tokens: List[Token]
    relations: List[Relation]

class DocumentsList(BaseModel):
    items: List[Document]

# === ГЕНЕРАЦИЯ ТЕКСТА НА ОСНОВЕ ЭВРИСТИК ===
def generate_text(tokens: List[Token], relations: List[Relation]) -> str:
    # Создаём словари для быстрого доступа
    token_map = {token.token_start: [[], token.text] for token in tokens}
    label_map = {token.token_start: token.entityLabel for token in tokens}

    # Склеиваем атрибуты с их компонентами
    new_relations = []
    for rel in relations:
        if rel.relationLabel == "ATTRIBUTE-FOR":
            attribute_name = token_map[rel.head][1]
            token_map[rel.child][0].append(attribute_name)
        else:
            new_relations.append(rel)

    relations = new_relations

    # Формируем новые имена: [атрибуты] + имя
    new_token_map = {}
    for i in token_map:
        if len(token_map[i][0]) > 0:
            new_token_map[i] = ", ".join(token_map[i][0]) + " " + token_map[i][1]
        else:
            new_token_map[i] = token_map[i][1]

    token_map = new_token_map

    # Построение графа для определения уровней
    G = nx.DiGraph()
    for rel in new_relations:
        G.add_edge(rel.child, rel.head)

    depths = {}
    try:
        for node in G.nodes:
            paths = list(nx.all_simple_paths(G, source=node, target=None))
            depths[node] = max(len(p) for p in paths) - 1 if paths else 0
    except:
        depths = {node: 0 for node in G.nodes}

    # Группировка связей по child
    grouped_relations = defaultdict(list)
    for rel in new_relations:
        grouped_relations[rel.child].append(rel)

    # Сортировка групп по глубине child
    sorted_groups = sorted(grouped_relations.items(), key=lambda x: depths.get(x[0], float('inf')))

    # Формирование текста
    result = []
    for child, rels in sorted_groups:
        group_lines = []
        for rel in rels:
            head_idx = rel.head
            rel_type = rel.relationLabel
            head_text = token_map.get(head_idx, f"[{head_idx}]")
            child_text = token_map.get(child, f"[{child}]")

            if rel_type == "PART-OF":
                line = f"{child_text} содержит {head_text}."
            elif rel_type == "ATTRIBUTE-FOR":
                line = f"{head_text} {child_text}."
            elif rel_type == "LOCATED-AT":
                line = f"{head_text} находится {child_text}."
            elif rel_type == "CONNECTED-WITH":
                line = f"{head_text} связан с {child_text}."
            elif rel_type == "IN-MANNER-OF":
                line = f"{child_text} выполнен в виде {head_text}."
            else:
                line = f"{child_text} [{rel_type}] {head_text}."

            group_lines.append(line)
        result.extend(group_lines)

    return "\n".join(result)


# === ВИЗУАЛИЗАЦИЯ ГРАФА ===
def visualize_relations_gv(tokens: List[Token], relations: List[Relation], output_path: str):
    color_map = {
        "SYSTEM": "#ffddba",
        "PART": "#c7f0db",
        "ATTRIBUTE": "#ffd1dc",
        "LOCATION": "#e3ffc7",
        "MANNER": "#ccccff",
        "OTHER": "#eeeeee"
    }

    dot = graphviz.Digraph(format='png', engine='dot')
    for token in tokens:
        label = f"{token.text} ({token.entityLabel})"
        color = color_map.get(token.entityLabel, color_map["OTHER"])
        dot.node(str(token.token_start), label, style='filled', fillcolor=color)

    for rel in relations:
        head = str(rel.head)
        child = str(rel.child)
        label = rel.relationLabel
        dot.edge(child, head, label=label)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dot.render(output_path, cleanup=True)


# === ИНТЕГРАЦИЯ С GPT-4 (через g4f) ===
def correct_text_with_gpt(prompt: str):
    client = Client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Вы — помощник, который переписывает технический текст в "
                                              "юридически структурированный формат."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            web_search=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Ошибка при подключении к GPT:", e)
        return None


# === ИНТЕГРАЦИЯ С Qwen2.5 ===
def correct_text_with_qwen(prompt: str):
    headers = {'Content-Type': 'application/json'}
    json_data = {
        'model': 'qwen2.5-7b-instruct-1m',
        'messages': [
            {'role': 'system', 'content': 'Вы — помощник, который переписывает технический текст в '
                                          'юридически структурированный формат.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.6,
        'max_tokens': -1,
        'stream': False
    }

    try:
        response = requests.post('http://localhost:1234/v1/chat/completions', headers=headers, json=json_data)
        if response.status_code == 200:
            corrected_text = response.json()['choices'][0]['message']['content']
            return corrected_text.strip()
        else:
            print(f"Ошибка запроса к LLM: {response.status_code}")
            return None
    except Exception as e:
        print("Ошибка подключения к Qwen:", e)
        return None


# === ИНТЕГРАЦИЯ С DeepSeek ===
def correct_text_with_deepseek(prompt: str):
    headers = {'Content-Type': 'application/json'}
    json_data = {
        'model': 'deepseek-r1-distill-llama-8b',
        'messages': [
            {'role': 'system', 'content': 'Вы — помощник, который переписывает технический текст '
                                          'в юридически структурированный формат.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.6,
        'max_tokens': -1,
        'stream': False
    }

    try:
        response = requests.post('http://localhost:1234/v1/chat/completions', headers=headers, json=json_data)
        if response.status_code == 200:
            corrected_text = response.json()['choices'][0]['message']['content'].strip()

            # удаление части текста, начинающейся с "<think>"
            corrected_text = corrected_text.split("</think>", 1)[-1].strip()
            return corrected_text
        else:
            print(f"Ошибка запроса к LLM: {response.status_code}")
            return None
    except Exception as e:
        print("Ошибка подключения к DeepSeek:", e)
        return None


# === ================== ===
if __name__ == "__main__":
    choice = input(
        "Нажмите:\n"
        "1 — если хотите получить текст, сгенерированный GPT-4 (необходимо подключение к интернету)\n"
        "2 — если хотите получить текст, сгенерированный Qwen2.5 (необходимо запустить локальный сервер)\n"
        "3 — если хотите получить текст, сгенерированный DeepSeek (необходимо запустить локальный сервер)\n"
        "Введите число: "
    )

    if choice not in ("1", "2", "3"):
        while True:
            choice = input(
                "Неверный выбор. Выберите:\n"
                "1 — если хотите получить текст, сгенерированный GPT-4 (необходимо подключение к интернету)\n"
                "2 — если хотите получить текст, сгенерированный Qwen2.5 (необходимо запустить локальный сервер)\n"
                "3 — если хотите получить текст, сгенерированный DeepSeek (необходимо запустить локальный сервер)\n"
                "Введите число: "
            )
            if choice in ("1", "2", "3"):
                break

    file_path = r'C:\Users\nazar\PycharmProjects\DIPLOM\relations_train.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    documents_list = DocumentsList(items=raw_data)
    documents = documents_list.items

    output_suffix = {
        "1": "1",
        "2": "2",
        "3": "3"
    }[choice]

    full_text = ""
    for i, doc in enumerate(documents):
        print(f"\n----------------------------------------ДОКУМЕНТ {i + 1}----------------------------------")

        original_text = doc.document
        print(f"Исходный текст:\n{original_text}\n")

        generated_text = generate_text(doc.tokens, doc.relations)
        print(f"Сгенерированный текст на основе эвристических правил:\n{generated_text}\n")

        # Визуализация графа
        output_file = f"./output/document_{i + 1}_graph"
        visualize_relations_gv(doc.tokens, doc.relations, output_file)
       # print(f"Граф сохранен в ./output/document_{i + 1}_graph.png")

        # Коррекция текста
        prompt = (
            "Перепиши следующий текст в более связный и юридически структурированный текст на русском языке.\n"
            "Сделай его подходящим под раздел 'формула изобретения' в патентной заявке.\n"
            "Cделай его в виде одного абзаца. Не добавляй дополнительных слов.\n"
            f"{generated_text}"
        )

        if choice == "1":
            corrected_text = correct_text_with_gpt(prompt)
            filename = "generated_text_1.txt"
        elif choice == "2":
            corrected_text = correct_text_with_qwen(prompt)
            filename = "generated_text_2.txt"
        else:
            corrected_text = correct_text_with_deepseek(prompt)
            filename = "generated_text_3.txt"

        print(f"Текст, скорректированный нейросетью:\n{corrected_text or 'Не удалось получить ответ от модели'}\n")
        print(f"Результат будет сохранен в {filename}")

        full_text += f"----------------------------------------ДОКУМЕНТ {i + 1}----------------------------------\n"
        full_text += f"Исходный текст:\n{original_text}\n\n"
        full_text += f"Сгенерированный текст на основе эвристических правил:\n{generated_text}\n\n"
        full_text += f"Текст, скорректированный нейросетью:\n{corrected_text}\n\n"

    with open(f"output/{filename}", "w", encoding="utf-8") as out_file:
        out_file.write(full_text)


