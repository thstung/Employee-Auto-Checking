import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="dcmm"
)

cursor = mydb.cursor()
sql = """
    create table EMPLOYEE(
        name CHAR(30),
        id INT,
        email_address VARCHAR(30),

    )
"""
cursor.execute()
