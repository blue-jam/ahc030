#!/usr/bin/python3
import csv
import os
import subprocess

# Compile Kotlin code
compile_command = "g++ -std=c++17 -O2 main.cpp"
os.system(compile_command)

# Initialize sum of outputs
total_sum = 0

run_command = f'seq 0 999 | xargs printf "%04d\n" | xargs -I@ -P8 sh -c "CI=true gtimeout 3 tools/target/debug/tester ./a.out < tools/in/@.txt > out/@.txt"'
os.system(run_command)
print("Outputs generated")


# accumulate the sum of outputs based on N, M into 2-dimensional list scores
scores = [[0 for i in range(21)] for j in range(21)]
count = [[0 for i in range(21)] for j in range(21)]
for i in range(0, 500):
    input_file = f"tools/in/{i:04d}.txt"
    output_file = f"out/{i:04d}.txt"

    with open(input_file, 'r') as file:
        first_line = file.readline().strip()

    Nf, Mf, e = map(float, first_line.split())
    N = int(Nf)
    M = int(Mf)

    command = f'tools/target/debug/vis {input_file} {output_file}'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(i, stdout.decode('utf-8'))

    score = int(stdout.decode('utf-8').split(' ')[2])
    scores[N][M] += score
    count[N][M] += 1

    total_sum += score

# output the average score for each N, M as table into csv file
with open('scores.csv', 'w') as file:
    file.write('N\\M,' + ','.join([str(i) for i in range(2, 21)]) + '\n')
    for i in range(10, 21):
        file.write(str(i) + ',' + ','.join([str(int(scores[i][j] / count[i][j]) if count[i][j] != 0 else 0) for j in range(2, 21)]) + '\n')


# Print the total sum of outputs
print(f"{total_sum}")


def generate_html(prev_file, scores_file, output_file):
    # Read prev.csv and scores.csv
    with open(prev_file, 'r') as prev_f, open(scores_file, 'r') as scores_f:
        prev_reader = csv.reader(prev_f)
        scores_reader = csv.reader(scores_f)
        prev_data = list(prev_reader)
        scores_data = list(scores_reader)

    # Create HTML table content
    html_content = "<table border='1'>\n"
    for i in range(len(scores_data)):
        html_content += "<tr>\n"
        if i == 0:
            html_content += "<th>N\\M</th>\n"
            for j in range(2, len(scores_data[i]) + 1):
                html_content += f"<th>{j}</th>\n"
            html_content += "</tr>\n"
        else:
            for j in range(len(scores_data[i])):
                prev_value = float(prev_data[i][j])
                score_value = float(scores_data[i][j])
                difference = abs(score_value - prev_value)
                if score_value < prev_value:
                    color_intensity = int((prev_value - score_value) * 10)
                    html_content += f"<td style='background-color: rgba(0, 255, 0, 0.{color_intensity})'>{score_value}</td>\n"
                elif score_value > prev_value:
                    color_intensity = int((score_value - prev_value) * 10)
                    html_content += f"<td style='background-color: rgba(255, 0, 0, 0.{color_intensity})'>{score_value}</td>\n"
                else:
                    html_content += f"<td>{score_value}</td>\n"
        html_content += "</tr>\n"
    html_content += "</table>"

    # Write HTML content to file
    with open(output_file, 'w') as output:
        output.write(html_content)

generate_html('prev.csv', 'scores.csv', 'output.html')
