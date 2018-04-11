#include <iostream>
#include<fstream>
#include <vector>
#include <algorithm>
#include <Eigen/Eigen> // <Eigen/Dense> for dense only, <Eigen/Sparse> for sparse only, <Eigen/Eigen> for both
#include <set>

using Mat = Eigen::MatrixXd;

std::set<std::string> parse_requested_column_names(std::string columns_names_to_read, char separator=';') {
    std::istringstream    iss(columns_names_to_read);
    std::set<std::string> columns_names_set;
    std::string           column_name;

    while (std::getline(iss, column_name, separator)) {
        columns_names_set.insert(column_name);
    }

    return columns_names_set;
}

std::vector<size_t> parse_column_names(std::string header_names_raw,
                                        std::string columns_names_to_read="",
                                        char separator = ';') {
    std::vector<std::string> header_names;
    std::vector<size_t> header_numbers;


    if (columns_names_to_read.empty()) {
        std::istringstream iss(header_names_raw);

        size_t idx = 0;
        std::string column_name;
        
        while (std::getline(iss, column_name, separator)) {
            header_names.push_back(column_name);
            header_numbers.push_back(idx);
            idx++;
        }

    } else {
        std::set<std::string> columns_names_set = parse_requested_column_names(columns_names_to_read);
        std::istringstream iss(header_names_raw);

        size_t idx = 0;
        std::string column_name;

        while (std::getline(iss, column_name, separator)) {
            if (columns_names_set.find(column_name) != columns_names_set.end()) {
                header_names.push_back(column_name);
                header_numbers.push_back(idx);
            }
            idx++;
        }
    }

    std::cout << "Read clumns: ";
    for (size_t i=0; i < header_names.size(); i++) {
        std::cout << header_names[i] << ":" << header_numbers[i] << " ";
    }
    std::cout << std::endl;

    return header_numbers;

}


template<class T> 
void print_function(T object) {
    for(auto it = object.begin(); it != object.end(); it++) {
        std::cout << *it << ", ";
    }
    std::cout << std::endl;
}

bool read_table(Mat* Y, std::string csv_file_to_read,
                std::string columns_names_to_read="",
                char separator=';') {

    std::ifstream infile(csv_file_to_read);
    std::string   line;
    
    std::getline(infile, line);
    std::vector<size_t> header_numbers = parse_column_names(line, columns_names_to_read, separator);
    std::set<size_t>    header_numbers_set(header_numbers.begin(), header_numbers.end());

    // Mat Y(header_numbers_set.size(), 0);
    Y->conservativeResize(header_numbers_set.size(), 0);

    while (std::getline(infile, line)) {
        std::istringstream iss(line);

        size_t column_idx = 0;
        size_t write_idx = 0;
        std::string element;
        Y->conservativeResize(Y->rows(), Y->cols() + 1);

        while (std::getline(iss, element, separator)) {
            if (header_numbers_set.find(column_idx) != header_numbers_set.end()) {
                (*Y)(write_idx, Y->cols()-1) = std::stod(element);
                write_idx++;
            }
            column_idx++;
        }
    }

    return true;
}

int main() {

    std::string csv_file_to_read     = "/Users/macintosh/Desktop/Current/SDA/Cryptocurrency-Forecast/data/sample/BTC-1ST_day_1.csv";
    char        separator = ';';
    std::string columns_names_to_read = "TU;O;C;";

    Mat Y;
    read_table(&Y, csv_file_to_read, columns_names_to_read, separator);
    std::cout << "Mat Y" << std::endl << Y << std::endl;
    return 0;

    Mat F(4, 2);
    F << 4,2,
         3,1,
         6,2,
         0,2;

    std::cout << "Mat F" << std::endl << F << std::endl;

    return 0;
}


