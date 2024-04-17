find ./ -type f \( -name "*.cu" -o -name "*.h" \) -exec clang-format -i {} +
