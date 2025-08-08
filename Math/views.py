from django.shortcuts import render
from google import genai
from google.genai import types
from django.http import JsonResponse
from openai import OpenAI
import requests
from groq import Groq
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import google.generativeai
import time
import plotly.graph_objs as plotly
import numpy as np
from .mathFunctions import log_base, sanitize_user_input


def graph(request):
    question = request.POST.get("question")
    google.generativeai.configure(api_key="AIzaSyC8xjiFVJ_7nwRoGk7nPHay6z8mdnpjYgQ")
    client = genai.Client()
    function = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
            get the function in: {question}
        
            Instructions:
            -Supported Operations:
              --Arithmetic: +, -, *, /, ** (exponent)...note:Use * explicitly for multiplication (write 2*x, not 2x)
              --Trigonometry: sin(x), cos(x), tan(x)
              --Inverse trig: asin(x), acos(x), atan(x)
              --Logarithms:
              --Natural log: log(x) or ln(x)
              --Base 10: log10(x)
              --Any base: logb(x, base) → (e.g. logb(x, 2) for log base 2)
              --Roots: sqrt(x)
              --Exponentials: exp(x)
              --Powers: x**2, pow(x, 3)
              --Absolute value: abs(x)
            - Constants:
              --Use pi for π (≈ 3.14)
              --Use e for Euler's number (≈ 2.718)

            - just get the fuction in it so i can use it in a variable and grapgh it 
            - convert it to text if it is not
            - don't return any thing in latyx
            - don't return it as y = func return the function only
            - if the fuction is in terms of y make it in terms i x
            - don't add + c
            - don't return a number beside x without putting the operation between them which will be '*'
            - Do not include any greetings, explanations, or non-answer text.
            """,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
        ),
    )
    print("Gemini Output:", repr(function.text))
    sanitize_user_input(raw_func=function.text)
    

    func_text = function.text.strip()

    try:
        x = np.linspace(-10, 10, 500)
        # Evaluate the function safely
        y = eval(
            func_text,
            {
                "x": x,
                "np": np,  # allow full numpy if needed
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "asin": np.arcsin,
                "acos": np.arccos,
                "atan": np.arctan,
                "sinh": np.sinh,
                "cosh": np.cosh,
                "tanh": np.tanh,
                "exp": np.exp,
                "log": np.log,  # natural log
                "ln": np.log,  # alias
                "log10": np.log10,  # log base 10
                "logb": log_base,  # log with arbitrary base
                "sqrt": np.sqrt,
                "abs": np.abs,
                "pow": np.power,
                "pi": np.pi,
                "e": np.e,
            },
        )
    except Exception as e:
        return {f"plot": "<p>Could not evaluate function.{e}</p>"}

    fig = plotly.Figure(data=plotly.Scatter(x=x, y=y, mode="lines"))
    fig.update_layout(
        title="Generated Plot",
        xaxis=dict(title="x", showgrid=True),
        yaxis=dict(title="y", showgrid=True),
    )

    pl_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    return pl_html


def home(request):
    pl_html = graph(request)

    return render(request, "home.html", {"plot": pl_html})


def update_graph(request):
    if request.method == "POST":
        pl_html = graph(request)
        return JsonResponse({"plot": pl_html})
    return JsonResponse({"error": "Invalid request"})


def send_gemeni(request):
    if request.method == "POST":

        question = request.POST.get("question")
        google.generativeai.configure(api_key="AIzaSyC8xjiFVJ_7nwRoGk7nPHay6z8mdnpjYgQ")
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
                Solve: {question}
                
                Instructions:
                - If the problem is complex, break it down step by step.
                - use LaTeX format and return this format in Markdown format using $$...$$ for any math dont write any thing with $...$. 
                - don't return any thing with $...$ remove them and then return what inside as text.
                - Put each step on a new line.
                - Write the final answer clearly on a new line at the end.
                - If the question is easy, just give the final answer directly.
                - Do not include any greetings, explanations, or non-answer text.
                """,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0
                )  # Disables thinking
            ),
        )

        return JsonResponse({"response": response.text})


def send_deepseek(request):
    question = request.POST.get("question")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {
                "role": "user",
                "content": f"""
                Solve: {question}
                
                Instructions:
                - If the problem is complex, break it down step by step.
                - use LaTeX format and return this format in Markdown format using $$...$$ for any math dont write any thing with $...$. 
                - don't return any thing with $...$ remove them and then return what inside as text.
                - Put each step on a new line.
                - Write the final answer clearly on a new line at the end.
                - If the question is easy, just give the final answer directly.
                - Do not include any greetings, explanations, or non-answer text.
                """,
            }
        ],
    )
    return JsonResponse({"response": completion.choices[0].message.content})


def send_welfram(request):
    question = request.POST.get("question")
    response = requests.get(
        f"http://api.wolframalpha.com/v2/query?appid=69PA493L7Q&input={question}&format=plaintext,mathml,images&output=json"
    )
    data = response.json()
    result = []
    for pod in data["queryresult"].get("pods", []):
        title = pod.get("title", "").lower()
        for subpod in pod.get("subpods", []):
            plaintext = subpod.get("plaintext", "")
            if plaintext:
                # If it's a step or result, show the title
                if any(
                    word in title
                    for word in [
                        "step",
                        "steps," "result",
                        "solution",
                        "answer",
                        "definite",
                        "indefinite",
                    ]
                ):
                    result.append(f"{pod['title']}:\n$${plaintext}$$")
                elif (
                    "input" not in title
                    and "plot" not in title
                    and "visual" not in title
                ):
                    # Show other useful text responses
                    result.append(f"{plaintext}")
    google.generativeai.configure(api_key="AIzaSyC8xjiFVJ_7nwRoGk7nPHay6z8mdnpjYgQ")
    client = genai.Client()
    readable_asnwer = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
            Format the following math solution which text for readability: {result}.
            
            Instructions:
            - Add proper spacing.
            - Put each step on a new line if there were steps. 
            - Wrap all math expressions (like equations, terms, square roots, fractions, etc.) in $$...$$ so they are rendered as LaTeX..
            - Do not add any AI comments, explanations, greetings, or extra content.
            - Only return the cleaned-up, formatted output.
            - Do not include any greetings or explanations
            """,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
        ),
    )
    final_output = (
        readable_asnwer.text if readable_asnwer.text else "No steps or result found."
    )

    return JsonResponse({"response": final_output})


def send_llama(requset):
    question = requset.POST.get("question")
    client = Groq(
        # This is the default and can be omitted
        api_key="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",
    )

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
                Solve: {question}
                
                Instructions:
                - If the problem is complex, break it down step by step.
                - put any between $$...$$ for any math don't write any thing with $...$. 
                - don't return any thing with $...$ remove them and then return what inside as text.
                - Put each step on a new line.
                - Write the final answer clearly on a new line at the end.
                - If the question is easy, just give the final answer directly.
                - Do not include any greetings, explanations, or non-answer text.
                - Do not add any unnecessary text.
                """,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    final_answer = f"{completion.choices[0].message.content}"

    return JsonResponse({"response": final_answer})


def send_sympolab(request):
    question_original = request.POST.get("question")
    google.generativeai.configure(api_key="AIzaSyC8xjiFVJ_7nwRoGk7nPHay6z8mdnpjYgQ")
    client = genai.Client()
    question = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
                just convert the math to laTeX: {question_original}
                
                Instructions:
                - i will use your response in a very sensitve url so make it perfect
                - don't return any thing with $...$ 
                - keep any text untouched just equations and formulas should be converted to laTeX
                - remove '/' from the final output becuase i use it in a url
                - Do not include any greetings or explanations just convert it without saying or adding a word from you
                """,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
        ),
    )
    url = f"https://en.symbolab.com/solver/step-by-step/{question.text}?or=input"
    with sync_playwright() as sym:
        browser = sym.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        
        page.wait_for_selector(".solution_math", timeout=15000)
        page.wait_for_selector(".chat-pane", timeout=10000)

        html_text = page.content()
        soub = BeautifulSoup(html_text, "lxml")
        answer_div = soub.find("div", class_="solution_math")
        answer_span = answer_div.find("span")
        answer = f"$$ {answer_span.get('title')} $$"
        steps = f"{soub.find('div',class_ = 'chat-pane').text}"
        browser.close()
    google.generativeai.configure(api_key="AIzaSyC8xjiFVJ_7nwRoGk7nPHay6z8mdnpjYgQ")
    client = genai.Client()
    readable_steps = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
            Format the following math solution which text for readability: {steps}.
            
            Instructions:
            - Add proper spacing.
            - Put each step on a new line. 
            - Wrap all math expressions (like equations, terms, square roots, fractions, etc.) in $$...$$ so they are rendered as LaTeX..
            - Do not add any AI comments, explanations, greetings, or extra content.
            - Only return the cleaned-up, formatted output.
            - Do not include any greetings or explanations
            """,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
        ),
    )
    final_answer = f"steps: {readable_steps.text if readable_steps else "No steps found."} \n final answer: {answer if answer else "No result found."}"
    return JsonResponse({"response": final_answer})


def send_stackexchange(request):
    soub = None
    question = request.POST.get("question")
    url = "https://api.stackexchange.com/2.3/search/advanced"

    params = {
        "order": "desc",
        "sort": "relevance",
        "q": question,
        "site": "math.stackexchange",
        "key": "rl_1AuonSCQ7azCQ8bcVfCL1gwkJ",  # Replace with your actual API key
    }
    response = requests.get(url, params=params)
    data = response.json()

    for item in data["items"][:8]:
        title = "title: " + item["title"]
        id = item["question_id"]
        if f"{question}" in item["title"]:
            print(title)
            answer_url = f"https://api.stackexchange.com/2.3/questions/{id}/answers"

            params = {
                "order": "desc",
                "sort": "votes",
                "site": "math.stackexchange",
                "filter": "withbody",
                "key": "rl_1AuonSCQ7azCQ8bcVfCL1gwkJ",  # Replace with your actual API key
            }
            answer_response = requests.get(answer_url, params=params)
            answer_data = answer_response.json()
            for answer in answer_data["items"]:
                ans = answer["body"]
                soub = BeautifulSoup(ans, "lxml")

            break

        time.sleep(2)
    if soub is not None:
        final_answer = {soub.text if soub.text else "no answer"}
    else:
        final_answer = "no answer"

    google.generativeai.configure(api_key="AIzaSyC8xjiFVJ_7nwRoGk7nPHay6z8mdnpjYgQ")
    client = genai.Client()
    readable_answer = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
            Format the following math solution which text for readability: {final_answer}.
            
            Instructions:
            - Add proper spacing.
            - Put each step on a new line. 
            - Wrap all math expressions (like equations, terms, square roots, fractions, etc.) in $$ . $$ so they are rendered as LaTeX.
            - don't put any thing in $...$
            - Do not add any AI comments, explanations, greetings, or extra content.
            - Only return the cleaned-up, formatted output.
            - Do not include any greetings or explanations
            """,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
        ),
    )

    return JsonResponse({"response": readable_answer.text})


# Create your views here.
