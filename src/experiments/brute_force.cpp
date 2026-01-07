#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <cctype>
#include <chrono>

using namespace std;

/* ---------- 工具函数 ---------- */

// 去掉非数字字符（保留 - . e E）
bool is_number_char(char c) {
    return isdigit(c) || c == '-' || c == '.' || c == 'e' || c == 'E';
}

// 从字符串中提取所有 double
vector<double> extract_numbers(const string& s) {
    vector<double> nums;
    string cur;
    for (char c : s) {
        if (is_number_char(c)) {
            cur += c;
        } else if (!cur.empty()) {
            nums.push_back(stod(cur));
            cur.clear();
        }
    }
    if (!cur.empty())
        nums.push_back(stod(cur));
    return nums;
}

/* ---------- 主程序 ---------- */

int main() {
    ifstream fin("ising_coeffs.txt");
    if (!fin) {
        cerr << "Failed to open ising_coeffs.txt\n";
        return 1;
    }

    string line, content;
    while (getline(fin, line)) {
        content += line + "\n";
    }
    fin.close();

    // -------- 解析 h --------
    size_t h_pos = content.find("h =");
    size_t J_pos = content.find("J =");
    size_t C_pos = content.find("C =");

    string h_str = content.substr(h_pos, J_pos - h_pos);
    string J_str = content.substr(J_pos, C_pos - J_pos);
    string C_str = content.substr(C_pos);

    vector<double> h = extract_numbers(h_str);
    vector<double> J_flat = extract_numbers(J_str);
    vector<double> C_vec = extract_numbers(C_str);

    if (C_vec.empty()) {
        cerr << "Failed to read C\n";
        return 1;
    }
    double C = C_vec[0];

    int N = h.size();
    if ((int)J_flat.size() != N * N) {
        cerr << "J size mismatch\n";
        return 1;
    }

    // 重构 J 矩阵
    vector<vector<double>> J(N, vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            J[i][j] = J_flat[i * N + j];

    cout << "Read N = " << N << endl;
    // printf("h = [ ");
    // for (double val : h)
    //     printf("%.4f ", val);
    // printf("]\n");
    // printf("J = [\n");
    // for (int i = 0; i < N; ++i) {
    //     printf("  [ ");
    //     for (int j = 0; j < N; ++j) {
    //         printf("%.4f ", J[i][j]);
    //     }
    //     printf("]\n");
    // }
    // printf("C = %.4f\n", C);

    // -------- 枚举所有 Ising 状态 --------

    double min_energy = numeric_limits<double>::infinity();
    double max_energy = -numeric_limits<double>::infinity();
    vector<int> min_state(N);

    long long total_states = 1LL << N;
    auto t_start = std::chrono::high_resolution_clock::now();


    for (long long state = 0; state < total_states; ++state) {
        if (state % 10000000 == 0) {
            cout << "Processing state " << state << " / " << total_states << "\r";
            cout.flush();
        }
        vector<int> z(N);

        // Python: z[i] = +1 if bit==0 else -1
        for (int i = 0; i < N; ++i) {
            z[i] = ((state >> i) & 1) == 0 ? +1 : -1;
        }

        double energy = C;

        // z^T J z
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                energy += z[i] * J[i][j] * z[j];

        // h^T z
        for (int i = 0; i < N; ++i)
            energy += h[i] * z[i];

        if (energy < min_energy) {
            min_energy = energy;
            min_state = z;
        }
        if (energy > max_energy) {
            max_energy = energy;
        }
    }

    // -------- 输出结果 --------

    cout << "Min energy: " << min_energy << endl;
    cout << "Min energy state z = [ ";
    for (int i = 0; i < N; ++i)
        cout << min_state[i] << " ";
    cout << "]" << endl;

    cout << "Max energy: " << max_energy << endl;
    
    // print total execution time
    auto t_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = t_end - t_start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";



    return 0;
}
