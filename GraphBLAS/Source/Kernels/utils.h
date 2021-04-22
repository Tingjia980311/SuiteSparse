__device__ int binarySearch(const int64_t* array,
                              int64_t        target,
                              int64_t        begin,
                              int64_t        end) {
  while (begin < end) {
    int mid  = begin + (end - begin) / 2;
    int item = array[mid];
    if (item == target)
      return mid;
    bool larger = (item > target);
    if (larger)
      end = mid;
    else
      begin = mid + 1;
  }
  return -1;
}

