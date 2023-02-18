@echo off
java -jar -Xmx1024m -Xms1024m target/gpt2-demo-1.0-jar-with-dependencies.jar model=SMALL maxLength=20

