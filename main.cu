#include <iostream>
#include <memory>
#include <vector>

#include <mpi.h>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    auto comm = MPI_COMM_WORLD;
    int rank;
    int size;

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    thrust::host_vector<double> host_local_buffer(10, static_cast<double>(rank));
    thrust::host_vector<double> host_shared_buffer(rank == 0 ? size * host_local_buffer.size() : 0, static_cast<double>(-1));

    thrust::device_vector<double> local_buffer = host_local_buffer;
    thrust::device_vector<double> shared_buffer = host_shared_buffer;

    MPI_Gather(local_buffer.data().get(), local_buffer.size(), MPI_DOUBLE,
               shared_buffer.data().get(), local_buffer.size(), MPI_DOUBLE, 0, comm);

    host_shared_buffer = shared_buffer;
    if(rank == 0) {
        std::cout << host_shared_buffer.size() << std::endl;
        for (auto i: shared_buffer) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
    MPI_Barrier(comm);

    std::cout << host_shared_buffer.size() << std::endl;

    MPI_Finalize();
}
