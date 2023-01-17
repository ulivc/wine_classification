from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def student_circuit_13():
    student_circuit = QuantumCircuit(3)

    phi1 = Parameter("phi1")
    phi2 = Parameter("phi2")
    phi3 = Parameter("phi3")
    phi4 = Parameter("phi4")
    phi5 = Parameter("phi5")
    phi6 = Parameter("phi6")
    phi7 = Parameter("phi7")
    phi8 = Parameter("phi8")
    phi9 = Parameter("phi9")
    phi10 = Parameter("phi10")
    phi11 = Parameter("phi11")
    phi12 = Parameter("phi12")

    student_circuit.ry(phi1, 0)
    student_circuit.ry(phi2, 1)

    student_circuit.ry(phi3, 2)

    student_circuit.crz(phi4, 2, 0)

    student_circuit.crz(phi5, 1, 2)
    student_circuit.crz(phi6, 0, 1)

    student_circuit.ry(phi7, 0)
    student_circuit.ry(phi8, 1)
    student_circuit.ry(phi9, 2)

    student_circuit.crz(phi10, 2, 1)
    student_circuit.crz(phi11, 0, 2)
    student_circuit.crz(phi12, 1, 0)

    return student_circuit


def student_circuit_14():
    student_circuit = QuantumCircuit(3)

    phi1 = Parameter("phi1")
    phi2 = Parameter("phi2")
    phi3 = Parameter("phi3")
    phi4 = Parameter("phi4")
    phi5 = Parameter("phi5")
    phi6 = Parameter("phi6")
    phi7 = Parameter("phi7")
    phi8 = Parameter("phi8")
    phi9 = Parameter("phi9")
    phi10 = Parameter("phi10")
    phi11 = Parameter("phi11")
    phi12 = Parameter("phi12")

    student_circuit.ry(phi1, 0)
    student_circuit.ry(phi2, 1)
    student_circuit.ry(phi3, 2)

    student_circuit.crx(phi4, 2, 0)

    student_circuit.crx(phi5, 1, 2)
    student_circuit.crx(phi6, 0, 1)

    student_circuit.ry(phi7, 0)
    student_circuit.ry(phi8, 1)
    student_circuit.ry(phi9, 2)

    student_circuit.crx(phi10, 2, 1)
    student_circuit.crx(phi11, 0, 2)
    student_circuit.crx(phi12, 1, 0)

    return student_circuit


def student_circuit_5():
    student_circuit = QuantumCircuit(3)

    phi1 = Parameter("phi1")
    phi2 = Parameter("phi2")
    phi3 = Parameter("phi3")
    phi4 = Parameter("phi4")
    phi5 = Parameter("phi5")
    phi6 = Parameter("phi6")
    phi7 = Parameter("phi7")
    phi8 = Parameter("phi8")
    phi9 = Parameter("phi9")
    phi10 = Parameter("phi10")
    phi11 = Parameter("phi11")
    phi12 = Parameter("phi12")
    phi13 = Parameter("phi13")
    phi14 = Parameter("phi14")
    phi15 = Parameter("phi15")
    phi16 = Parameter("phi16")
    phi17 = Parameter("phi17")

    student_circuit.rx(phi1, 0)
    student_circuit.rx(phi2, 1)
    student_circuit.rx(phi3, 2)

    student_circuit.rz(phi3, 0)
    student_circuit.rz(phi4, 1)
    student_circuit.rz(phi5, 2)

    student_circuit.crz(phi6, 2, 0)
    student_circuit.crz(phi7, 2, 1)

    student_circuit.crz(phi8, 1, 2)
    student_circuit.crz(phi9, 1, 0)

    student_circuit.crz(phi10, 0, 2)
    student_circuit.crz(phi11, 0, 1)

    student_circuit.rx(phi12, 0)
    student_circuit.rx(phi13, 1)
    student_circuit.rx(phi14, 2)

    student_circuit.rz(phi15, 0)
    student_circuit.rz(phi16, 1)
    student_circuit.rz(phi17, 2)

    return student_circuit


def student_circuit_6():
    student_circuit = QuantumCircuit(3)

    phi1 = Parameter("phi1")
    phi2 = Parameter("phi2")
    phi3 = Parameter("phi3")
    phi4 = Parameter("phi4")
    phi5 = Parameter("phi5")
    phi6 = Parameter("phi6")
    phi7 = Parameter("phi7")
    phi8 = Parameter("phi8")
    phi9 = Parameter("phi9")
    phi10 = Parameter("phi10")
    phi11 = Parameter("phi11")
    phi12 = Parameter("phi12")
    phi13 = Parameter("phi13")
    phi14 = Parameter("phi14")
    phi15 = Parameter("phi15")
    phi16 = Parameter("phi16")
    phi17 = Parameter("phi17")

    student_circuit.rx(phi1, 0)
    student_circuit.rx(phi2, 1)
    student_circuit.rx(phi3, 2)

    student_circuit.rz(phi3, 0)
    student_circuit.rz(phi4, 1)
    student_circuit.rz(phi5, 2)

    student_circuit.crx(phi6, 2, 0)
    student_circuit.crx(phi7, 2, 1)

    student_circuit.crx(phi8, 1, 2)
    student_circuit.crx(phi9, 1, 0)

    student_circuit.crx(phi10, 0, 2)
    student_circuit.crx(phi11, 0, 1)

    student_circuit.rx(phi12, 0)
    student_circuit.rx(phi13, 1)
    student_circuit.rx(phi14, 2)

    student_circuit.rz(phi15, 0)
    student_circuit.rz(phi16, 1)
    student_circuit.rz(phi17, 2)

    return student_circuit
