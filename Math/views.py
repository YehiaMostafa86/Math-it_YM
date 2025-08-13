from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
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
import mimetypes
latyx_image = ''


def image2latyx(request):
    response = None
    image = request.FILES.get('image')
    print('yehiaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    if image :
        print('workedddddddddddddddddddddddddddddddddd')
        
        image_bytes = image.read()
        mimetype, _ = mimetypes.guess_type(image.name)
        
        if not mimetype:
            if image.name.lower().endswith(('.png','.PNG')):
                mimetype = 'image/png'
            elif image.name.lower().endswith(('.jpg','.jpeg','.JPG','.JPEG')):
                mimetype = 'image/jpeg'
            elif image.name.lower().endswith(('.gif','.GIF')):
                mimetype = 'image/gif'
            elif image.name.lower().endswith(('.webp','.WEBp')):
                mimetype = 'image/webp'
            else:
                mimetype = 'image/jpeg'
            
    
        google.generativeai.configure(api_key=f'{settings.GEMINI_API_KEY}')
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                """convert this image to Latyx. You are a math OCR tool. Read the mathematical expressions from this image and return only the LaTeX code for them. 
                   Do not explain, do not add extra text, and do not include delimiters like $$ or \[ \]. 
                   Only output valid LaTeX.""",
                types.Part.from_bytes(
                    data=image_bytes, mime_type=mimetype or "application/octet-stream"
                ),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
            ),
        )
    
        latyx_image = open('image_latyx.txt','w')
        latyx_image.write(response.text)
        latyx_image.close()
    else:
        print('deletedddddddddddddddffff')
        latyx_image = open('image_latyx.txt','w')
        latyx_image.write('')
        latyx_image.close()
        
    return JsonResponse({"message": "function worked"})



def graph(request):
    question = request.POST.get("question",'')  

    image_read = open('image_latyx.txt','r')
    latyx_image = image_read.read()
        

    
    google.generativeai.configure(api_key=f'{settings.GEMINI_API_KEY}')
    client = genai.Client()
    function = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
            get the function in: {question} or {latyx_image} (see which of them has the func)
        
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
        question = request.POST.get("question",'')
        
        image_read = open('image_latyx.txt','r')
        latyx_image = image_read.read()
        
   
        
        google.generativeai.configure(api_key=f'{settings.GEMINI_API_KEY}')
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
                Solve: {question} and {latyx_image}
                
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
    question = request.POST.get("question",'')
    image_read = open('image_latyx.txt','r')
    latyx_image = image_read.read()
    
  

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        
        api_key=f"{settings.DEEPSEEK_API_KEY}",
    )

    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {
                "role": "user",
                "content": f"""
                Solve: {question} and {latyx_image}
                
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
    question = request.POST.get("question",'')
    image_read = open('image_latyx.txt','r')
    latyx_image = image_read.read()
    
    
    response = requests.get(
        f"http://api.wolframalpha.com/v2/query?appid={settings.WELFRAM_API_KEY}&input={question}{latyx_image}&format=plaintext,mathml,images&output=json"
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
    google.generativeai.configure(api_key=f'{settings.GEMINI_API_KEY}')
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


def send_llama(request):
    question = request.POST.get("question",'')
    image_read = open('image_latyx.txt','r')
    latyx_image = image_read.read()


    client = Groq(
        # This is the default and can be omitted
        api_key=f"{settings.LLAMA_API_KEY}",
    )

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
                Solve: {question} and {latyx_image}
                
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
    question_original = request.POST.get("question",'')
    image_read = open('image_latyx.txt','r')
    latyx_image = image_read.read()

    google.generativeai.configure(api_key=f'{settings.GEMINI_API_KEY}')
    client = genai.Client()
    question = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
                just convert the math to laTeX: {question_original}, {latyx_image} 
                
                Instructions:
                - combine the them to be a single math problem in latyx
                - don't sove the question just convert
                - i will use your response in a very sensitve url so make it perfect
                - remove $..$ and dont write $..$ or $$..$$
                - keep any text untouched just equations and formulas should be converted to laTeX
                - remove '/' from the final output becuase i use it in a url
                - Do not include any greetings or explanations just convert it without saying or adding a word from you
                """,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
        ),
    )
    print(question.text)
    url = f"https://en.symbolab.com/solver/step-by-step/{question.text}?or=input"
    with sync_playwright() as sym:
        browser = sym.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        page.wait_for_selector(".solution_math", timeout=20000)
        page.wait_for_selector(".chat-pane", timeout=20000)

        html_text = page.content()
        soub = BeautifulSoup(html_text, "lxml")
        answer_div = soub.find("div", class_="solution_math")
        answer_span = answer_div.find("span")
        answer = f"$$ {answer_span.get('title')} $$"
        steps = f"{soub.find('div',class_ = 'chat-pane').text}"
        browser.close()
    google.generativeai.configure(api_key=f'{settings.GEMINI_API_KEY}')
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
    image_read = open('image_latyx.txt','r')
    latyx_image = image_read.read()

    url = "https://api.stackexchange.com/2.3/search/advanced"

    params = {
        "order": "desc",
        "sort": "relevance",
        "q": f'{question} {latyx_image}',  
        "site": "math.stackexchange",
        "key": f"{settings.STACKEXCHANGE_API_KEY}",  # Replace with your actual API key
    }
    response = requests.get(url, params=params)
    data = response.json()

    for item in data["items"][:8]:
        title = "title: " + item["title"]
        id = item["question_id"]
        if f"{question}" in item["title"]:
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

    google.generativeai.configure(api_key=f'{settings.GEMINI_API_KEY}')
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
