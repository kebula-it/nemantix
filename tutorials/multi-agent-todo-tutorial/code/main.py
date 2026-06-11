import sqlite3
from pathlib import Path
from nemantix.core.tools import Toolset, tool
from nemantix.core import Expertise, Agent
from nemantix.security import Verifier

class TodoManagerToolset(Toolset):

    def __init__(self, db_uri: str = "todos.db"):
        super().__init__()
        self._db = sqlite3.connect(db_uri)
        self._init_db()

    def __del__(self):
        self.close()

    def close(self):
        self._db.close()

    def _init_db(self):
        cu = self._db.cursor()

        cu.execute('''
        CREATE TABLE IF NOT EXISTS Todo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            is_completed INT DEFAULT 0
        )
        ''')

        self._db.commit()
        cu.close()

    @tool
    def create_todo(self, text: str) -> bool:
        cu = self._db.cursor()

        try:
            cu.execute('INSERT INTO Todo (text) VALUES (?)', (text,))
            self._db.commit()
            return True
        except sqlite3.Error as e:
            print(f"Todo Creation error: {e}")
            return False
        finally:
            cu.close()        

    @tool
    def find_todo(self, todo_id: int) -> dict:
        cu = self._db.cursor()

        cu.execute(
            'SELECT id, text, is_completed, created_at FROM Todo WHERE id = ?',
            (todo_id,)
        )

        row = cu.fetchone()
        cu.close()
        
        if not row:
            return {
                "status": "error",
                "error": f"No Todo found with ID {todo_id}"
            }

        return {
            "status": "success",
            "todo": {
                'id': row[0],
                'text': row[1],
                'is_completed': bool(row[2]),
            }
        }

    @tool
    def list_todos(self) -> dict:
        cu = self._db.cursor()
        cu.execute('SELECT id, text, is_completed, created_at FROM Todo')
        rows = cu.fetchall()
        cu.close()
        
        todos = [
            {
                'id': row[0],
                'text': row[1],
                'is_completed': bool(row[2]),
            }
            for row in rows
        ]
        
        return {"status": "success", "todos": todos}

    @tool
    def delete_todo(self, todo_id: int) -> bool:
        cu = self._db.cursor()
        cu.execute('DELETE FROM Todo WHERE id = ?', (todo_id,))
        rows_affected = cu.rowcount
        self._db.commit()
        cu.close()
        
        return rows_affected > 0

    @tool
    def complete_todo(self, todo_id: int) -> bool:

        cu = self._db.cursor()
        cu.execute('UPDATE Todo SET is_completed = 1 WHERE id = ?', (todo_id,))
        rows_affected = cu.rowcount
        self._db.commit()
        cu.close()
        
        return rows_affected > 0

    @tool
    def flush_todos(self) -> dict:
        cu = self._db.cursor()
        cu.execute('DELETE FROM Todo WHERE is_completed = 1')
        deleted_count = cu.rowcount
        self._db.commit()
        cu.close()
        
        return {"status": "success", "deleted_count": deleted_count}

def main() -> None:

    current_folder = Path.cwd()

    verifier = Verifier(current_folder / 'keys/publickey.crt')
    credentials = current_folder / 'credentials.json'

    reader_exp = Expertise.from_local_scripts(
        paths=[current_folder / 'nxs/reader.nxs'],
        verifier=verifier,
        credentials_path=credentials,
    )

    writer_exp = Expertise.from_local_scripts(
        paths=[current_folder / 'nxs/writer.nxs'],
        verifier=verifier,
        credentials_path=credentials,
    )

    reader_agent = Agent(expertise=reader_exp, build_on_start=True)
    writer_agent = Agent(expertise=writer_exp, build_on_start=True)

    while True:
        print(
            """
            === Agentic TODO Manager ===
            new      - Creates a new todo
            list     - List all todos
            find     - Find a todo by id
            delete   - Delete todo by id
            complete - Mark a todo as completed
            flush    - Remove all completed todos
            exit     - Exit
            """
        )

        command = input(": ").strip().lower()

        if command == 'exit':
            break
            
        elif command == 'new':
            task = input("Enter the task description: ")
            err, out = writer_agent.run(user_request=f"Create a new todo with this description: {task}")
        elif command == 'list':
            err, out = reader_agent.run(user_request="List all my todos")
        elif command == 'find':
            todo_id = input("Enter the todo ID: ")
            err, out = reader_agent.run(user_request=f"Find the todo with id: {todo_id}")
        elif command == 'delete':
            todo_id = input("Enter the ID of the todo to delete: ")
            err, out = writer_agent.run(user_request=f"Delete the todo with id: {todo_id}")
        elif command == 'complete':
            todo_id = input("Enter the ID of the todo to complete: ")
            err, out = writer_agent.run(user_request=f"Mark the todo with id {todo_id} as completed")
        elif command == 'flush':
            err, out = writer_agent.run(user_request="Execute flush removing all completed todos")
        else:
            print(f"Unrecognized command: '{command}'")
            continue

        if err:
            print(f"\n[AGENT ERROR]: {err}")
        else:
            print(f"\n[AGENT RESPONSE]: {out}")

if __name__ == '__main__':
    main()
