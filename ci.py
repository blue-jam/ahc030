#!/usr/bin/python3
import os
import subprocess

# Compile Kotlin code
compile_command = "g++ -std=c++17 main.cpp"
os.system(compile_command)

# Initialize sum of outputs
total_sum = 0

run_command = f'seq 0 99 | xargs printf "%04d\n" | xargs -I@ -P8 sh -c "CI=true time tools/target/debug/tester ./a.out < tools/in/@.txt > out/@.txt"'
os.system(run_command)
print("Outputs generated")
for i in range(0, 100):
    input_file = f"tools/in/{i:04d}.txt"
    output_file = f"out/{i:04d}.txt"

    command = f'tools/target/debug/vis {input_file} {output_file}'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(i, stdout.decode('utf-8'))
    total_sum += int(stdout.decode('utf-8').split(' ')[2])

# Print the total sum of outputs
print(f"{total_sum}")