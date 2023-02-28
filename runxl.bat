@echo off
CHCP 65001
java -Xmx7G -Xms7G -jar target/gpt2-demo-1.0-jar-with-dependencies.jar model=XL maxLength=10

