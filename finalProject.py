import os, csv, json, time
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Header, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic


# Step1: Load environment variables and set API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

APP_KEY = os.environ.get("KEY_API_APP")
ANTHROPIC_KEY = os.environ.get("KEY_API_ANTHROPIC")

TASKS_FILE = "tasks.json"
AI_LOGS_FILE = "ai_logs.csv"

cache: Dict[str, Any] = {"users_df": None, "tasks_list": []}
conversations: Dict[str, List[Dict[str, str]]] = {}


# Step2: Fetch JSON data from a given path with error handling
def fetch_json(path: str) -> Optional[Any]:
    try:
        res = requests.get(f"https://jsonplaceholder.typicode.com/{path.lstrip('/')}", timeout=5)
        res.raise_for_status()
        return res.json()
    except requests.RequestException as e:
        print(f"Error fetching {path}: {e}")
        return None

json_fetch = fetch_json


def init_cache():
    users = fetch_json("users")
    todos = fetch_json("todos")
    if not users or not todos:
        return bool(cache["users_df"] is not None)

    # DataFrame for users with selected fields
    cache["users_df"] = pd.DataFrame([
        {"user_id": u["id"], "name": u["name"], "email": u["email"], "city": u.get("address", {}).get("city", "")}
        for u in users
    ])

    # Load tasks from file if exists, otherwise initialize from todos and save
    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            cache["tasks_list"] = json.load(f)
    except FileNotFoundError:
        todos_df = pd.DataFrame(todos).rename(columns={"userId": "user_id"})
        cache["tasks_list"] = [
            {"id": t["id"], "title": t["title"], "user_id": t["user_id"], "done": bool(t["completed"])}
            for t in todos_df.to_dict(orient="records")
        ]
        save_tasks()
    return True


def save_tasks():
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(cache["tasks_list"], f, indent=2, ensure_ascii=False)


# Step3: Analyze data (Pandas & Numpy)
def get_analysis():
    if cache["users_df"] is None and not init_cache():
        raise HTTPException(status_code=503, detail="Data source unavailable")

    tasks_df = pd.DataFrame(cache["tasks_list"])
    tasks_df["completed"] = tasks_df["done"].astype(bool)
    merged = pd.merge(tasks_df, cache["users_df"], on="user_id", how="inner")

    # groupby and aggregate to calculate total tasks, completed tasks, and completion rate per user
    grouped = merged.groupby(["user_id", "name", "email", "city"]).agg(
        total_tasks=("completed", "count"),
        completed_tasks=("completed", lambda x: int(np.sum(x)))
    ).reset_index()
    grouped["completion_rate"] = np.round((grouped["completed_tasks"] / grouped["total_tasks"]) * 100, 2)
    users_sorted = grouped.sort_values(by="completion_rate", ascending=False).to_dict(orient="records")

    rates = [u["completion_rate"] for u in users_sorted]
    mean_val = float(np.round(np.mean(rates), 2)) if rates else 0.0
    std_val = float(np.round(np.std(rates), 2)) if rates else 0.0

    # boolean indexing for low performers (completion rate below mean)
    low_performers = grouped[grouped["completion_rate"] < mean_val].to_dict(orient="records")

    # pivot_table for city-wise completed and uncompleted tasks
    pivot = pd.pivot_table(merged, index="city", columns="completed", values="user_id", aggfunc="count", fill_value=0)
    cities, strongest_city, max_done = [], "N/A", -1
    for city in pivot.index:
        done = int(pivot.loc[city, True]) if True in pivot.columns else 0
        undone = int(pivot.loc[city, False]) if False in pivot.columns else 0
        if done > max_done:
            max_done, strongest_city = done, city
        cities.append({"city": city, "completed": done, "uncompleted": undone, "total": done + undone})

    return {
        "users": users_sorted, "mean": mean_val, "std": std_val,
        "low_performers": low_performers, "cities": cities, "strongest_city": strongest_city
    }


# Step4: Define models and dependencies
class TaskIn(BaseModel):
    title: str
    user_id: int
    done: bool = False

class TaskOut(BaseModel):
    id: int
    title: str
    user_id: int
    done: bool

class AskRequest(BaseModel):
    prompt: str
    conversation_id: str

class TaskNotFoundError(Exception):
    def __init__(self, task_id: int):
        self.task_id = task_id


# Initialize cache and FastAPI app
init_cache()
app = FastAPI(title="Team Performance API")

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def process_time_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Process-Time"] = f"{duration:.4f}s"
    print(f"[{request.method}] {request.url.path} - Status: {response.status_code} - {duration:.4f}s")
    return response

@app.exception_handler(TaskNotFoundError)
async def task_not_found_handler(request: Request, exc: TaskNotFoundError):
    return JSONResponse(status_code=404, content={"detail": f"Task {exc.task_id} not found", "task_id": exc.task_id})

def check_auth(x_api_key: Optional[str] = Header(None, alias="X-API-Key"), key_api_x: Optional[str] = Header(None, alias="Key-API-X")):
    key = x_api_key or key_api_x
    if not key or key != APP_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return key

def paginate(items: list, page: int = 1, page_size: int = 10):
    start = (page - 1) * page_size
    return {"page": page, "page_size": page_size, "total_items": len(items), "items": items[start:start + page_size]}

def find_task(task_id: int) -> dict:
    for t in cache["tasks_list"]:
        if t["id"] == task_id:
            return t
    raise TaskNotFoundError(task_id)

def validate_task(task: TaskIn):
    if not task.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    if cache["users_df"] is not None and task.user_id not in cache["users_df"]["user_id"].values:
        raise HTTPException(status_code=400, detail=f"User {task.user_id} does not exist")


# Step5: Define routes for stats and tasks
@app.get("/stats/summary")
def get_summary():
    data = get_analysis()
    total_tasks = sum(u["total_tasks"] for u in data["users"])
    total_done = sum(u["completed_tasks"] for u in data["users"])
    return {
        "total_users": len(data["users"]),
        "total_tasks": total_tasks,
        "completed_tasks": total_done,
        "overall_completion_rate": round(total_done / total_tasks * 100, 2) if total_tasks else 0,
        "mean_completion_rate": data["mean"],
        "std_completion_rate": data["std"],
        "strongest_city": data["strongest_city"],
        "top_performer": data["users"][0] if data["users"] else None
    }

@app.get("/stats/users")
def get_users_stats(sort_by: str = "completion_rate", order: str = "desc", page: int = Query(1, ge=1), page_size: int = Query(10, ge=1)):
    users = get_analysis()["users"]
    if sort_by in ["total_tasks", "completed_tasks", "completion_rate", "name"]:
        users = sorted(users, key=lambda x: x.get(sort_by, 0), reverse=(order.lower() == "desc"))
    return paginate(users, page, page_size)

@app.get("/stats/low-performers")
def get_low_performers_route(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1)):
    return paginate(get_analysis()["low_performers"], page, page_size)

@app.get("/stats/cities")
def get_cities_route():
    a = get_analysis()
    return {"cities": a["cities"], "strongest_city": a["strongest_city"]}


# Step6: Define routes for tasks CRUD with validation
@app.get("/tasks/overdue")
def get_overdue_tasks(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1)):
    return paginate([], page, page_size)

@app.get("/tasks")
def list_tasks(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1)):
    return paginate(cache["tasks_list"], page, page_size)

@app.post("/tasks", response_model=TaskOut, status_code=201)
def create_task(task: TaskIn, _auth=Depends(check_auth)):
    validate_task(task)
    new_id = max([t["id"] for t in cache["tasks_list"]], default=0) + 1
    new_task = {"id": new_id, "title": task.title.strip(), "user_id": task.user_id, "done": task.done}
    cache["tasks_list"].append(new_task)
    save_tasks()
    return TaskOut(**new_task)

@app.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: int):
    return TaskOut(**find_task(task_id))

@app.put("/tasks/{task_id}", response_model=TaskOut)
def update_task(task_id: int, task: TaskIn, _auth=Depends(check_auth)):
    validate_task(task)
    t = find_task(task_id)
    t.update({"title": task.title.strip(), "user_id": task.user_id, "done": task.done})
    save_tasks()
    return TaskOut(**t)

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int, _auth=Depends(check_auth)):
    t = find_task(task_id)
    cache["tasks_list"].remove(t)
    save_tasks()
    return {"message": f"Task {task_id} deleted"}


# Step7: Define routes for AI interactions with logging
def log_ai(prompt_len: int, status_str: str):
    if not os.path.exists(AI_LOGS_FILE):
        with open(AI_LOGS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "prompt_length", "status"])
    try:
        with open(AI_LOGS_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.now().isoformat(), prompt_len, status_str])
    except Exception:
        pass

@app.post("/ai/ask")
def ask_ai(req: AskRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(req.prompt) > 500:
        raise HTTPException(status_code=400, detail="Prompt exceeds 500 characters")

    analysis = get_analysis()
    top_3 = ", ".join([u["name"] for u in analysis["users"][:3]])
    system_prompt = f"Team avg: {analysis['mean']}%. Top: {top_3}. Strongest city: {analysis['strongest_city']}."

    status_code = "error"
    try:
        if ANTHROPIC_KEY and ANTHROPIC_KEY != "123":
            try:
                client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
                msgs = conversations.get(req.conversation_id, []) + [{"role": "user", "content": req.prompt}]
                res = client.messages.create(model="claude-3-haiku-20240307", max_tokens=300, system=system_prompt, messages=msgs)
                reply = res.content[0].text
                conversations.setdefault(req.conversation_id, []).extend([{"role": "user", "content": req.prompt}, {"role": "assistant", "content": reply}])
                status_code = "success"
                return {"conversation_id": req.conversation_id, "answer": reply}
            except Exception:
                pass

        reply = f"[Mock AI] Team avg: {analysis['mean']}%. Top performers: {top_3}. Strongest city: {analysis['strongest_city']}."
        conversations.setdefault(req.conversation_id, []).extend([{"role": "user", "content": req.prompt}, {"role": "assistant", "content": reply}])
        status_code = "success"
        return {"conversation_id": req.conversation_id, "answer": reply}
    finally:
        log_ai(len(req.prompt), status_code)

@app.get("/stats/logs/ai")
@app.get("/ai/logs/stats")
def ai_logs_stats():
    if not os.path.exists(AI_LOGS_FILE):
        return {"total_calls": 0, "success_rate": 0.0, "average_prompt_length": 0.0}
    with open(AI_LOGS_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"total_calls": 0, "success_rate": 0.0, "average_prompt_length": 0.0}
    successes = sum(1 for r in rows if r.get("status") == "success")
    lengths = [int(r.get("prompt_length", 0)) for r in rows if r.get("prompt_length", "").isdigit()]
    return {
        "total_calls": len(rows),
        "success_rate": round((successes / len(rows)) * 100, 2),
        "average_prompt_length": round(sum(lengths) / len(lengths), 2) if lengths else 0.0
    }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("finalProject:app", host="127.0.0.1", port=8000, reload=True)
