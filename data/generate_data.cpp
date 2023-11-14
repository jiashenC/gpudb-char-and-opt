#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

void uniform_random(std::vector<int> &vec, int num, int max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distr(1, max);

  for (int i = 0; i < num; i++) {
    vec.push_back(distr(gen));
  }
}

void flush_txt(std::vector<int> &k_vec, std::vector<int> &v_vec) {
  std::ofstream of;
  of.open("temp.txt", std::ios::out);

  of << "pk,attr\n";
  for (int i = 0; i < k_vec.size(); i++) {
    of << k_vec[i] << "," << v_vec[i] << "\n";
  }
  of.close();
}

void flush_bin(std::vector<int> &k_vec, std::vector<int> &v_vec) {
  std::ofstream of0;
  of0.open("temp0.bin", std::ios::out | std::ios::binary);
  of0.write(reinterpret_cast<char *>(&k_vec[0]), k_vec.size() * sizeof(int));
  of0.close();

  std::ofstream of1;
  of1.open("temp1.bin", std::ios::out | std::ios::binary);
  of1.write(reinterpret_cast<char *>(&v_vec[0]), v_vec.size() * sizeof(int));
  of1.close();

  // load vector to verify
  // std::vector<int> in_vec(k_vec.size());
  // std::ifstream if0;
  // if0.open("temp0.bin", std::ios::in | std::ios::binary);
  // if0.read(reinterpret_cast<char *>(&in_vec[0]), in_vec.size() * sizeof(int));
  // if0.close();
  // for (int i = 0; i < in_vec.size(); i++) {
  //   assert(in_vec[i] == k_vec[i]);
  // }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "./generate_data <number of value> <max value> <distribution>"
              << std::endl;
    exit(1);
  }

  int num = std::stoi(argv[1]), max = std::stoi(argv[2]),
      dist = std::stoi(argv[3]);

  std::cout << "Generate data: " << num << " " << max << " " << dist
            << std::endl;

  std::vector<int> k_vec, v_vec;

  if (dist == 0) {
    uniform_random(k_vec, num, max);
    uniform_random(v_vec, num, max);
  }

  // std::cout << k_vec.size() << std::endl;

  flush_txt(k_vec, v_vec);
  flush_bin(k_vec, v_vec);
}