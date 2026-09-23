import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minidb import Database, Session

path = sys.argv[1]
mode = sys.argv[2]
db = Database(path)
s = Session(db)
if mode == "setup":
    s.sql("CREATE TABLE acct (id INT PRIMARY KEY, bal INT)")
    s.sql("INSERT INTO acct VALUES (1, 100), (2, 200)")
    import time; time.sleep(0.3)
    os._exit(0)  # hard exit WITHOUT shutdown/checkpoint (simulate crash)
elif mode == "commit_then_crash":
    s.sql("BEGIN")
    s.sql("UPDATE acct SET bal = bal - 50 WHERE id = 1")
    s.sql("UPDATE acct SET bal = bal + 50 WHERE id = 2")
    s.sql("COMMIT")
    os._exit(0)
elif mode == "mid_txn_crash":
    s.sql("BEGIN")
    s.sql("UPDATE acct SET bal = 999 WHERE id = 1")
    s.sql("INSERT INTO acct VALUES (3, 999)")
    os._exit(0)
