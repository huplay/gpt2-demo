@echo off
CHCP 65001
java -Xmx4G -Xms4G -jar target/gpt2-demo-1.0-jar-with-dependencies.jar model=LARGE maxLength=10

