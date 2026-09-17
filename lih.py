from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper
import numpy as np

# Step 1: Run PySCF driver without active space transformer
driver = PySCFDriver(
    atom="Li 0 0 0; H 2.5 0 0",
    unit=DistanceUnit.ANGSTROM,
    basis="sto3g"
)
es_problem_full = driver.run()

# Get the full electronic energy from PySCF (Hartree-Fock SCF energy)
full_scf_energy = es_problem_full.reference_energy
print(f"Full SCF energy (including all orbitals): {full_scf_energy}")

# Step 2: Apply ActiveSpaceTransformer
transformer = ActiveSpaceTransformer(2, 3, active_orbitals=[1, 2, 3])
es_problem_active = transformer.transform(es_problem_full)

# Step 3: Extract nuclear repulsion energy
nuclear_repulsion = es_problem_active.hamiltonian.nuclear_repulsion_energy

# Step 4: Get qubit operator and lowest eigenvalue
hamiltonian = es_problem_active.hamiltonian.second_q_op()
mapper = ParityMapper(num_particles=es_problem_active.num_particles)
qubit_op = mapper.map(hamiltonian)
eigvals = np.linalg.eigvalsh(qubit_op.to_matrix())
lowest_eigval = eigvals[0]
print(qubit_op.paulis)
print(qubit_op.coeffs)

# Step 5: Calculate frozen core energy (difference)
# Frozen core energy = full SCF energy - (active space electronic energy + nuclear repulsion)
frozen_core_energy = full_scf_energy - (lowest_eigval + nuclear_repulsion)

# Step 6: Calculate total energy
total_energy = lowest_eigval + nuclear_repulsion + frozen_core_energy

print(f"Lowest eigenvalue (active space): {lowest_eigval}")
print(f"Nuclear repulsion energy: {nuclear_repulsion}")
print(f"Frozen core energy (computed): {frozen_core_energy}")
print(f"Total ground state energy (corrected): {total_energy}")
