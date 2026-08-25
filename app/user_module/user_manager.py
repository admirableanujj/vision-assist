#app/user_module/user_manager.py
"""
User Manager Module

Handles user registration, secure password hashing, database persistence,
and session authentication against the PostgreSQL backend using your custom database schema.

Usage:
    from user_module.user_manager import UserManager
    user_mgr = UserManager()

Dependencies:
    psycopg2-binary==2.9.9
    
__original_author__ = "Anujj Saxena"    
"""
import os
import hashlib
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor

__author__ = "Anujj Saxena"
__license__ = "MIT"
__version__ = "1.0.2"

class UserManager:
    def __init__(self):
        self.db_host = "vision_assist_db"
        self.db_name = os.getenv("POSTGRES_DB", "vision_assist")
        self.db_user = os.getenv("POSTGRES_USER", "postgres")
        self.db_password = os.getenv("POSTGRES_PASSWORD", "")

    def _get_connection(self):
        return psycopg2.connect(
            host=self.db_host,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )

    def _hash_password(self, password: str, salt: str) -> str:
        """Generates a secure SHA-256 hash incorporating the salt."""
        # Because your custom schema does not hold a separate salt column, 
        # we will embed the salt internally inside the SHA-256 generation chain.
        salted_pass = password.encode('utf-8') + salt.encode('utf-8')
        hashed = hashlib.sha256(salted_pass).hexdigest()
        # Combine salt and hash into one string to persist in user_login.hashed_password
        return f"{salt}:{hashed}"

    def _verify_password(self, password: str, stored_credential: str) -> bool:
        """Verifies an incoming password against the saved stored salt & hash combination."""
        try:
            salt, stored_hash = stored_credential.split(":")
            recalculated_hash = hashlib.sha256(password.encode('utf-8') + salt.encode('utf-8')).hexdigest()
            return recalculated_hash == stored_hash
        except ValueError:
            return False

    def register_user(self, username: str, password: str) -> tuple[bool, str]:
        """Registers a user profile and links credentials across both tables."""
        if not username or not password:
            return False, "Username and password cannot be empty."

        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                # 1. Create Profile in users table (Default standard user role: 1)
                guid = str(uuid.uuid4())
                user_insert = """
                INSERT INTO users (guid_user, username, role, is_active) 
                VALUES (%s, %s, 1, TRUE) RETURNING id;
                """
                cur.execute(user_insert, (guid, username))
                user_id = cur.fetchone()[0]

                # 2. Hash password and save credentials inside user_login table
                salt = uuid.uuid4().hex
                credential_string = self._hash_password(password, salt)
                
                login_insert = """
                INSERT INTO user_login (user_id, hashed_password)
                VALUES (%s, %s);
                """
                cur.execute(login_insert, (user_id, credential_string))
                
            conn.commit()
            return True, "Registration successful!"
        except psycopg2.errors.UniqueViolation:
            if conn:
                conn.rollback()
            return False, "Username already exists."
        except Exception as e:
            if conn:
                conn.rollback()
            return False, f"Database write failure: {e}"
        finally:
            if conn:
                conn.close()

    def authenticate_user(self, username: str, password: str) -> bool:
        """Verifies coordinates against users join user_login query patterns."""
        query = """
        SELECT u.id, ul.hashed_password 
        FROM users u
        JOIN user_login ul ON u.id = ul.user_id
        WHERE u.username = %s AND u.is_active = TRUE;
        """
        conn = None
        user_id = None
        status = "failed"
        authenticated = False
        
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (username,))
                record = cur.fetchone()
                
                if record:
                    user_id = record['id']
                    if self._verify_password(password, record['hashed_password']):
                        status = "success"
                        authenticated = True
            return authenticated
        except Exception as e:
            print(f"[ERROR] Authentication pipeline exception: {e}")
            return False
        finally:
            # Audit logins inside your user_login_history table
            if conn and user_id:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO user_login_history (user_id, login_status) VALUES (%s, %s)",
                            (user_id, status)
                        )
                    conn.commit()
                except Exception as audit_error:
                    print(f"[WARN] Failed to write audit record: {audit_error}")
            
            if conn:
                conn.close()