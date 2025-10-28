import numpy as np
import pygad
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum

# ========== CONFIGURACIÓN DEL HORARIO ==========
BLOQUES_TIEMPO = [
    "07:20-08:05", "08:05-08:50", "08:50-09:35", "09:35-10:20",
    "10:20-10:40",  # RECREO
    "10:40-11:25", "11:25-12:10", "12:10-12:55",
    "12:55-13:25",  # ALMUERZO
    "13:25-14:10", "14:10-14:55"
]

HORAS_PEDAGOGICAS = [
    "07:20-08:05", "08:05-08:50", "08:50-09:35", "09:35-10:20",
    "10:40-11:25", "11:25-12:10", "12:10-12:55",
    "13:25-14:10", "14:10-14:55"
]

RECREO_INDICE = 4  # Índice en BLOQUES_TIEMPO
ALMUERZO_INDICE = 8  # Índice en BLOQUES_TIEMPO
HORAS_POR_DIA = 9  # Horas pedagógicas por día (sin recreos)
NUM_DIAS = 5

class TipoBloque(Enum):
    """Tipos de bloques de clase por duración"""
    SIMPLE = 1      # 1 hora pedagógica (45 min)
    DOBLE = 2       # 2 horas pedagógicas (90 min)
    TRIPLE = 3      # 3 horas pedagógicas (135 min)

@dataclass
class Docente:
    """Representa un docente con sus características"""
    id: int
    nombre: str
    tipo: str  # "nombrado" o "contratado"
    horas_semanales_requeridas: int
    disponibilidad: np.ndarray  # Shape: (5 días, 9 horas pedagógicas)
    materias: List[str]  # Materias que puede enseñar

@dataclass
class Grado:
    """Representa un grado escolar (sin secciones)"""
    id: int
    numero: int  # 3, 4, 5, etc.
    num_secciones: int  # Cantidad de secciones (A, B, C...)
    
    @property
    def nombre(self):
        return f"Grado {self.numero}"
    
    def get_seccion_nombre(self, seccion_idx: int) -> str:
        """Obtiene el nombre de una sección específica"""
        if seccion_idx >= self.num_secciones:
            raise ValueError(f"Sección {seccion_idx} no existe en grado {self.numero}")
        return f"{self.numero}{chr(65 + seccion_idx)}"  # 3A, 3B, 3C...

@dataclass
class BloqueClase:
    """
    Representa un bloque de clase: una o más horas consecutivas 
    de una materia específica con un docente para un grado-sección
    """
    id: int
    docente_id: int
    grado_id: int
    seccion_idx: int  # Índice de la sección (0=A, 1=B, etc.)
    materia: str
    tipo_bloque: TipoBloque  # SIMPLE, DOBLE o TRIPLE
    
    @property
    def duracion(self) -> int:
        """Retorna la duración en horas pedagógicas"""
        return self.tipo_bloque.value


class AlgoritmoGeneticoHorarios:
    """
    Algoritmo genético mejorado para generación de horarios escolares
    
    Estructura del cromosoma:
    - Para cada grado-sección: array de 45 posiciones (9 horas × 5 días)
    - Cada posición = ID del bloque de clase asignado (o -1 si vacío)
    - Cromosoma total = concatenación de todos los grados-secciones
    """
    
    def __init__(self, 
                 docentes: List[Docente], 
                 grados: List[Grado], 
                 bloques: List[BloqueClase],
                 num_generaciones: int = 150,
                 tamano_poblacion: int = 100):
        
        # Almacenar entidades
        self.docentes = {d.id: d for d in docentes}
        self.grados = {g.id: g for g in grados}
        self.bloques = {b.id: b for b in bloques}
        
        # Calcular total de secciones
        self.secciones = []  # Lista de tuplas (grado_id, seccion_idx, nombre)
        for grado in grados:
            for sec_idx in range(grado.num_secciones):
                nombre = grado.get_seccion_nombre(sec_idx)
                self.secciones.append((grado.id, sec_idx, nombre))
        
        # Parámetros del algoritmo
        self.num_generaciones = num_generaciones
        self.tamano_poblacion = tamano_poblacion
        
        # Pesos para la función de fitness
        self.pesos = {
            'huecos': 50.0,           # Penaliza huecos entre clases
            'conflictos': 1000.0,     # Penaliza solapamientos
            'horas': 200.0,           # Penaliza exceso/déficit de horas
            'disponibilidad': 500.0,   # Penaliza violación de disponibilidad
            'distribucion': 30.0,      # Penaliza mala distribución semanal
            'continuidad': 20.0        # Bonifica bloques bien ubicados
        }
        
        # Configuración del cromosoma
        self.horas_por_seccion = HORAS_POR_DIA * NUM_DIAS  # 45
        self.longitud_cromosoma = self.horas_por_seccion * len(self.secciones)
        
        # Validar bloques
        self._validar_bloques()
        
        print(f"🎯 Sistema inicializado:")
        print(f"   - Docentes: {len(self.docentes)}")
        print(f"   - Grados: {len(self.grados)}")
        print(f"   - Secciones totales: {len(self.secciones)}")
        print(f"   - Bloques de clase: {len(self.bloques)}")
        print(f"   - Longitud cromosoma: {self.longitud_cromosoma}")
    
    def _validar_bloques(self):
        """Valida que los bloques estén correctamente configurados"""
        for bloque_id, bloque in self.bloques.items():
            # Verificar que el docente existe
            if bloque.docente_id not in self.docentes:
                raise ValueError(f"Bloque {bloque_id}: Docente {bloque.docente_id} no existe")
            
            # Verificar que el grado existe
            if bloque.grado_id not in self.grados:
                raise ValueError(f"Bloque {bloque_id}: Grado {bloque.grado_id} no existe")
            
            # Verificar que la sección existe en el grado
            grado = self.grados[bloque.grado_id]
            if bloque.seccion_idx >= grado.num_secciones:
                raise ValueError(
                    f"Bloque {bloque_id}: Sección {bloque.seccion_idx} "
                    f"no existe en grado {grado.numero} (máx: {grado.num_secciones-1})"
                )
    
    def indice_a_dia_hora(self, indice: int) -> Tuple[int, int]:
        """Convierte índice local (0-44) a (día, hora_pedagógica)"""
        dia = indice // HORAS_POR_DIA
        hora = indice % HORAS_POR_DIA
        return dia, hora
    
    def dia_hora_a_indice(self, dia: int, hora: int) -> int:
        """Convierte (día, hora_pedagógica) a índice local (0-44)"""
        return dia * HORAS_POR_DIA + hora
    
    def obtener_seccion_de_indice(self, indice_global: int) -> Tuple[int, int, str, int]:
        """
        Obtiene información de sección desde índice global del cromosoma
        Retorna: (grado_id, seccion_idx, nombre_seccion, indice_local)
        """
        seccion_idx = indice_global // self.horas_por_seccion
        indice_local = indice_global % self.horas_por_seccion
        grado_id, sec_idx, nombre = self.secciones[seccion_idx]
        return grado_id, sec_idx, nombre, indice_local
    
    def fitness_function(self, ga_instance, solution, solution_idx):
        """Función de fitness principal (a MAXIMIZAR en PyGAD)"""
        penalizacion_total = 0.0
        
        # Calcular cada componente de penalización
        penalizacion_total += self.pesos['huecos'] * self._calc_huecos(solution)
        penalizacion_total += self.pesos['conflictos'] * self._calc_conflictos(solution)
        penalizacion_total += self.pesos['horas'] * self._calc_horas(solution)
        penalizacion_total += self.pesos['disponibilidad'] * self._calc_disponibilidad(solution)
        penalizacion_total += self.pesos['distribucion'] * self._calc_distribucion(solution)
        
        # Bonificación por continuidad
        bonificacion = self.pesos['continuidad'] * self._calc_continuidad(solution)
        
        # PyGAD maximiza, así que retornamos el negativo de la penalización
        return -penalizacion_total + bonificacion
    
    def _calc_huecos(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza huecos entre clases del mismo docente en un día
        """
        penalizacion = 0.0
        
        for docente_id, _ in self.docentes.items():
            # Para cada día
            for dia in range(NUM_DIAS):
                horas_docente = []
                
                # Buscar todas las horas donde enseña este docente
                for idx_global in range(len(cromosoma)):
                    bloque_id = int(cromosoma[idx_global])
                    if bloque_id >= 0 and bloque_id in self.bloques:
                        bloque = self.bloques[bloque_id]
                        if bloque.docente_id == docente_id:
                            _, _, _, idx_local = self.obtener_seccion_de_indice(idx_global)
                            dia_clase, hora_clase = self.indice_a_dia_hora(idx_local)
                            if dia_clase == dia:
                                horas_docente.append(hora_clase)
                
                # Calcular huecos
                if len(horas_docente) > 1:
                    horas_docente.sort()
                    for i in range(len(horas_docente) - 1):
                        hueco = horas_docente[i + 1] - horas_docente[i] - 1
                        if hueco > 0:
                            penalizacion += hueco
        
        return penalizacion
    
    def _calc_conflictos(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza conflictos:
        1. Docente en dos lugares al mismo tiempo
        2. Bloque mal asignado (en sección incorrecta)
        3. Bloques superpuestos en la misma sección
        """
        penalizacion = 0.0
        
        # 1. Detectar docentes en múltiples lugares
        ocupacion_docente = {}  # (docente_id, dia, hora) -> lista de secciones
        
        for idx_global in range(len(cromosoma)):
            bloque_id = int(cromosoma[idx_global])
            if bloque_id >= 0 and bloque_id in self.bloques:
                bloque = self.bloques[bloque_id]
                grado_id, sec_idx, nombre_sec, idx_local = self.obtener_seccion_de_indice(idx_global)
                dia, hora = self.indice_a_dia_hora(idx_local)
                
                # Verificar asignación correcta del bloque
                if bloque.grado_id != grado_id or bloque.seccion_idx != sec_idx:
                    penalizacion += 100  # Penalización alta por bloque mal ubicado
                    continue
                
                key = (bloque.docente_id, dia, hora)
                if key not in ocupacion_docente:
                    ocupacion_docente[key] = []
                ocupacion_docente[key].append(nombre_sec)
        
        # Penalizar docentes en múltiples lugares
        for key, secciones in ocupacion_docente.items():
            if len(secciones) > 1:
                penalizacion += (len(secciones) - 1) * 50
        
        # 2. Detectar bloques superpuestos en la misma sección
        for sec_idx in range(len(self.secciones)):
            inicio = sec_idx * self.horas_por_seccion
            fin = inicio + self.horas_por_seccion
            seccion_cromosoma = cromosoma[inicio:fin]
            
            # Verificar que no haya IDs duplicados (excepto -1)
            bloques_asignados = [int(b) for b in seccion_cromosoma if int(b) >= 0]
            if len(bloques_asignados) != len(set(bloques_asignados)):
                # Hay bloques repetidos
                repeticiones = len(bloques_asignados) - len(set(bloques_asignados))
                penalizacion += repeticiones * 20
        
        return penalizacion
    
    def _calc_horas(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza diferencia entre horas asignadas y requeridas por docente
        """
        penalizacion = 0.0
        horas_asignadas = {d_id: 0 for d_id in self.docentes.keys()}
        
        # Contar horas asignadas por docente
        for idx_global in range(len(cromosoma)):
            bloque_id = int(cromosoma[idx_global])
            if bloque_id >= 0 and bloque_id in self.bloques:
                bloque = self.bloques[bloque_id]
                # Solo contar una vez por bloque (no por cada hora del bloque)
                # Verificar si es el inicio del bloque
                if idx_global == 0 or cromosoma[idx_global - 1] != bloque_id:
                    horas_asignadas[bloque.docente_id] += bloque.duracion
        
        # Calcular penalizaciones
        for docente_id, docente in self.docentes.items():
            diferencia = abs(horas_asignadas[docente_id] - docente.horas_semanales_requeridas)
            if diferencia > 0:
                penalizacion += diferencia ** 2  # Penalización cuadrática
        
        return penalizacion
    
    def _calc_disponibilidad(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza asignaciones en horarios no disponibles del docente
        """
        penalizacion = 0.0
        
        for idx_global in range(len(cromosoma)):
            bloque_id = int(cromosoma[idx_global])
            if bloque_id >= 0 and bloque_id in self.bloques:
                bloque = self.bloques[bloque_id]
                docente = self.docentes[bloque.docente_id]
                
                _, _, _, idx_local = self.obtener_seccion_de_indice(idx_global)
                dia, hora = self.indice_a_dia_hora(idx_local)
                
                if not docente.disponibilidad[dia, hora]:
                    penalizacion += 1
        
        return penalizacion
    
    def _calc_distribucion(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza mala distribución de materias durante la semana
        """
        penalizacion = 0.0
        
        # Para cada sección
        for sec_idx in range(len(self.secciones)):
            inicio = sec_idx * self.horas_por_seccion
            fin = inicio + self.horas_por_seccion
            seccion_cromosoma = cromosoma[inicio:fin]
            
            # Contar materias por día
            materias_por_dia = {dia: set() for dia in range(NUM_DIAS)}
            
            for idx_local, bloque_id in enumerate(seccion_cromosoma):
                bloque_id = int(bloque_id)
                if bloque_id >= 0 and bloque_id in self.bloques:
                    bloque = self.bloques[bloque_id]
                    dia, _ = self.indice_a_dia_hora(idx_local)
                    materias_por_dia[dia].add(bloque.materia)
            
            # Penalizar días con muchas materias diferentes
            for dia, materias in materias_por_dia.items():
                if len(materias) > 3:  # Más de 3 materias diferentes en un día
                    penalizacion += (len(materias) - 3) * 2
        
        return penalizacion
    
    def _calc_continuidad(self, cromosoma: np.ndarray) -> float:
        """
        Bonifica bloques bien ubicados (continuos y sin fragmentación)
        """
        bonificacion = 0.0
        
        # Verificar que bloques multi-hora estén continuos
        for sec_idx in range(len(self.secciones)):
            inicio = sec_idx * self.horas_por_seccion
            fin = inicio + self.horas_por_seccion
            seccion_cromosoma = cromosoma[inicio:fin]
            
            bloque_actual = -1
            horas_consecutivas = 0
            
            for idx_local, bloque_id in enumerate(seccion_cromosoma):
                bloque_id = int(bloque_id)
                
                if bloque_id == bloque_actual and bloque_id >= 0:
                    horas_consecutivas += 1
                else:
                    # Verificar si el bloque anterior estaba completo
                    if bloque_actual >= 0 and bloque_actual in self.bloques:
                        bloque_ant = self.bloques[bloque_actual]
                        if horas_consecutivas == bloque_ant.duracion:
                            bonificacion += 1  # Bloque completo y continuo
                    
                    bloque_actual = bloque_id
                    horas_consecutivas = 1 if bloque_id >= 0 else 0
        
        return bonificacion
    
    def crear_poblacion_inicial(self) -> np.ndarray:
        """
        Genera población inicial con heurísticas inteligentes
        """
        poblacion = []
        
        for _ in range(self.tamano_poblacion):
            cromosoma = -np.ones(self.longitud_cromosoma, dtype=int)
            
            # Intentar asignar cada bloque
            for bloque_id, bloque in self.bloques.items():
                docente = self.docentes[bloque.docente_id]
                grado = self.grados[bloque.grado_id]
                
                # Encontrar la sección correcta en el cromosoma
                sec_idx_global = None
                for i, (g_id, s_idx, _) in enumerate(self.secciones):
                    if g_id == bloque.grado_id and s_idx == bloque.seccion_idx:
                        sec_idx_global = i
                        break
                
                if sec_idx_global is None:
                    continue
                
                inicio_seccion = sec_idx_global * self.horas_por_seccion
                
                # Intentar colocar el bloque en posiciones válidas
                intentos = 0
                colocado = False
                
                while intentos < 20 and not colocado:
                    # Posición aleatoria que permita el bloque completo
                    max_pos = self.horas_por_seccion - bloque.duracion + 1
                    if max_pos > 0:
                        pos_local = np.random.randint(0, max_pos)
                        
                        # Verificar disponibilidad y espacio libre
                        valido = True
                        for offset in range(bloque.duracion):
                            idx_global = inicio_seccion + pos_local + offset
                            dia, hora = self.indice_a_dia_hora(pos_local + offset)
                            
                            # Verificar disponibilidad del docente
                            if not docente.disponibilidad[dia, hora]:
                                valido = False
                                break
                            
                            # Verificar que el espacio esté libre
                            if cromosoma[idx_global] != -1:
                                valido = False
                                break
                        
                        if valido:
                            # Asignar el bloque
                            for offset in range(bloque.duracion):
                                cromosoma[inicio_seccion + pos_local + offset] = bloque_id
                            colocado = True
                    
                    intentos += 1
            
            poblacion.append(cromosoma)
        
        return np.array(poblacion)
    
    def mutacion_personalizada(self, offspring, ga_instance):
        """
        Operador de mutación personalizado que respeta restricciones
        PyGAD requiere esta firma específica para mutaciones personalizadas
        """
        for chromosome_idx in range(offspring.shape[0]):
            if np.random.random() < 0.3:  # 30% de probabilidad de mutación
                cromosoma = offspring[chromosome_idx]
                
                # Elegir tipo de mutación
                tipo_mutacion = np.random.choice(['swap', 'move', 'remove'])
                
                if tipo_mutacion == 'swap':
                    # Intercambiar dos bloques del mismo grado-sección
                    sec_idx = np.random.randint(0, len(self.secciones))
                    inicio = sec_idx * self.horas_por_seccion
                    fin = inicio + self.horas_por_seccion
                    
                    # Encontrar dos bloques para intercambiar
                    indices_ocupados = [i for i in range(inicio, fin) if cromosoma[i] >= 0]
                    if len(indices_ocupados) >= 2:
                        idx1, idx2 = np.random.choice(indices_ocupados, 2, replace=False)
                        cromosoma[idx1], cromosoma[idx2] = cromosoma[idx2], cromosoma[idx1]
                
                elif tipo_mutacion == 'move':
                    # Mover un bloque a otra posición
                    indices_ocupados = [i for i in range(len(cromosoma)) if cromosoma[i] >= 0]
                    if indices_ocupados:
                        idx = np.random.choice(indices_ocupados)
                        bloque_id = cromosoma[idx]
                        
                        # Limpiar posición actual
                        cromosoma[idx] = -1
                        
                        # Buscar nueva posición válida
                        intentos = 0
                        while intentos < 10:
                            nuevo_idx = np.random.randint(0, len(cromosoma))
                            if cromosoma[nuevo_idx] == -1:
                                cromosoma[nuevo_idx] = bloque_id
                                break
                            intentos += 1
                
                elif tipo_mutacion == 'remove':
                    # Remover un bloque aleatorio (para reducir conflictos)
                    indices_ocupados = [i for i in range(len(cromosoma)) if cromosoma[i] >= 0]
                    if indices_ocupados and len(indices_ocupados) > 20:  # Solo si hay muchos bloques
                        idx = np.random.choice(indices_ocupados)
                        cromosoma[idx] = -1
        
        return offspring
    
    def ejecutar(self) -> Tuple[np.ndarray, pygad.GA]:
        """
        Ejecuta el algoritmo genético
        """
        print("\n🧬 Iniciando optimización genética...")
        print(f"   Población: {self.tamano_poblacion}")
        print(f"   Generaciones: {self.num_generaciones}")
        
        # Configurar espacio de genes
        gene_space = list(range(-1, len(self.bloques)))
        
        # Crear instancia de PyGAD
        ga_instance = pygad.GA(
            num_generations=self.num_generaciones,
            num_parents_mating=max(10, self.tamano_poblacion // 5),
            fitness_func=self.fitness_function,
            sol_per_pop=self.tamano_poblacion,
            num_genes=self.longitud_cromosoma,
            gene_space=gene_space,
            gene_type=int,
            initial_population=self.crear_poblacion_inicial(),
            parent_selection_type="tournament",
            K_tournament=5,
            crossover_type="uniform",
            crossover_probability=0.9,
            mutation_type="random",  # Usar mutación estándar de PyGAD
            mutation_probability=0.2,
            mutation_num_genes=3,  # Número de genes a mutar
            keep_elitism=max(2, self.tamano_poblacion // 20),
            suppress_warnings=True,
            save_best_solutions=True,
            # Callback para aplicar mutación personalizada adicional
            on_mutation=self.aplicar_restricciones_post_mutacion
        )
        
        # Ejecutar
        ga_instance.run()
        
        # Obtener mejor solución
        solution, solution_fitness, _ = ga_instance.best_solution()
        
        print(f"\n✅ Optimización completada!")
        print(f"   Fitness final: {solution_fitness:.2f}")
        
        # Mostrar desglose de penalizaciones
        self._mostrar_desglose_fitness(solution)
        
        return solution, ga_instance
    
    def aplicar_restricciones_post_mutacion(self, ga_instance, offspring):
        """
        Callback para aplicar restricciones después de la mutación estándar
        Corrige violaciones de restricciones introducidas por la mutación
        """
        for idx in range(offspring.shape[0]):
            cromosoma = offspring[idx]
            
            # Verificar y corregir bloques mal asignados
            for i in range(len(cromosoma)):
                bloque_id = int(cromosoma[i])
                if bloque_id >= 0 and bloque_id in self.bloques:
                    bloque = self.bloques[bloque_id]
                    grado_id, sec_idx, _, _ = self.obtener_seccion_de_indice(i)
                    
                    # Si el bloque está en la sección incorrecta, removerlo
                    if bloque.grado_id != grado_id or bloque.seccion_idx != sec_idx:
                        cromosoma[i] = -1
            
            # Eliminar duplicados de bloques (mantener solo la primera ocurrencia)
            bloques_vistos = set()
            for i in range(len(cromosoma)):
                bloque_id = int(cromosoma[i])
                if bloque_id >= 0:
                    if bloque_id in bloques_vistos:
                        cromosoma[i] = -1  # Remover duplicado
                    else:
                        bloques_vistos.add(bloque_id)
        
        return offspring
    
    def _mostrar_desglose_fitness(self, solucion: np.ndarray):
        """Muestra el desglose de penalizaciones de la solución"""
        print("\n📊 Desglose de penalizaciones:")
        print(f"   - Huecos: {self._calc_huecos(solucion):.1f}")
        print(f"   - Conflictos: {self._calc_conflictos(solucion):.1f}")
        print(f"   - Horas: {self._calc_horas(solucion):.1f}")
        print(f"   - Disponibilidad: {self._calc_disponibilidad(solucion):.1f}")
        print(f"   - Distribución: {self._calc_distribucion(solucion):.1f}")
        print(f"   - Continuidad (bonus): {self._calc_continuidad(solucion):.1f}")
    
    def generar_horarios(self, solucion: np.ndarray) -> Dict[str, pd.DataFrame]:
        """
        Genera DataFrames de horarios para cada sección
        """
        horarios = {}
        
        for sec_idx, (grado_id, seccion_idx, nombre_seccion) in enumerate(self.secciones):
            inicio = sec_idx * self.horas_por_seccion
            fin = inicio + self.horas_por_seccion
            seccion_cromosoma = solucion[inicio:fin]
            
            # Crear matriz del horario
            horario_matrix = np.empty((HORAS_POR_DIA, NUM_DIAS), dtype=object)
            horario_matrix[:] = ""
            
            # Rastrear bloques ya procesados
            bloques_procesados = set()
            
            for idx_local, bloque_id in enumerate(seccion_cromosoma):
                bloque_id = int(bloque_id)
                if bloque_id >= 0 and bloque_id in self.bloques:
                    bloque = self.bloques[bloque_id]
                    
                    # Verificar que el bloque pertenece a esta sección
                    if bloque.grado_id != grado_id or bloque.seccion_idx != seccion_idx:
                        continue  # Bloque mal asignado, ignorar
                    
                    docente = self.docentes[bloque.docente_id]
                    dia, hora = self.indice_a_dia_hora(idx_local)
                    
                    # Solo agregar información si es la primera hora del bloque
                    if bloque_id not in bloques_procesados:
                        info = f"{bloque.materia}\n{docente.nombre}\n({bloque.duracion}h)"
                        bloques_procesados.add(bloque_id)
                    else:
                        info = "↓"  # Indicador de continuación
                    
                    if horario_matrix[hora, dia] and horario_matrix[hora, dia] != "↓":
                        horario_matrix[hora, dia] += f"\n⚠️ CONFLICTO"
                    else:
                        horario_matrix[hora, dia] = info
            
            # Convertir a DataFrame
            df = pd.DataFrame(
                horario_matrix,
                index=HORAS_PEDAGOGICAS,
                columns=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
            )
            
            horarios[nombre_seccion] = df
        
        return horarios
    
    def imprimir_horarios(self, solucion: np.ndarray):
        """
        Imprime horarios formateados para cada sección
        """
        horarios = self.generar_horarios(solucion)
        
        print("\n" + "="*100)
        print("📚 HORARIOS OPTIMIZADOS POR SECCIÓN")
        print("="*100)
        
        for nombre_seccion, df in horarios.items():
            print(f"\n{'-'*100}")
            print(f"📖 SECCIÓN: {nombre_seccion}")
            print(f"{'-'*100}")
            print(df.to_string())
            
            # Estadísticas de la sección
            total_horas = (df != "").sum().sum()
            print(f"\n   Total horas asignadas: {total_horas}")
    
    def exportar_horarios(self, solucion: np.ndarray, archivo_base: str = "horario"):
        """
        Exporta los horarios a archivos Excel
        """
        horarios = self.generar_horarios(solucion)
        
        with pd.ExcelWriter(f"{archivo_base}_completo.xlsx") as writer:
            for nombre_seccion, df in horarios.items():
                df.to_excel(writer, sheet_name=nombre_seccion)
        
        print(f"\n💾 Horarios exportados a {archivo_base}_completo.xlsx")
    
    def generar_estadisticas_docentes(self, solucion: np.ndarray) -> pd.DataFrame:
        """
        Genera estadísticas de carga horaria por docente
        """
        stats = []
        
        for docente_id, docente in self.docentes.items():
            horas_asignadas = 0
            materias_dictadas = set()
            secciones_atendidas = set()
            
            for idx_global in range(len(solucion)):
                bloque_id = int(solucion[idx_global])
                if bloque_id >= 0 and bloque_id in self.bloques:
                    bloque = self.bloques[bloque_id]
                    if bloque.docente_id == docente_id:
                        # Contar solo una vez por bloque
                        if idx_global == 0 or solucion[idx_global - 1] != bloque_id:
                            horas_asignadas += bloque.duracion
                            materias_dictadas.add(bloque.materia)
                            grado = self.grados[bloque.grado_id]
                            seccion = grado.get_seccion_nombre(bloque.seccion_idx)
                            secciones_atendidas.add(seccion)
            
            stats.append({
                'Docente': docente.nombre,
                'Tipo': docente.tipo,
                'Horas Requeridas': docente.horas_semanales_requeridas,
                'Horas Asignadas': horas_asignadas,
                'Diferencia': horas_asignadas - docente.horas_semanales_requeridas,
                'Materias': ', '.join(materias_dictadas),
                'Secciones': ', '.join(sorted(secciones_atendidas))
            })
        
        return pd.DataFrame(stats)


# ========== EJEMPLO DE USO ==========
if __name__ == "__main__":
    
    # Configurar disponibilidad de docentes
    disponibilidad_completa = np.ones((NUM_DIAS, HORAS_POR_DIA), dtype=bool)
    
    disponibilidad_parcial = np.ones((NUM_DIAS, HORAS_POR_DIA), dtype=bool)
    disponibilidad_parcial[0, 5:] = False  # Lunes tarde no disponible
    disponibilidad_parcial[3, 5:] = False  # Jueves tarde no disponible
    
    # Crear docentes
    docentes = [
        Docente(
            id=0, 
            nombre="Edwin García", 
            tipo="nombrado", 
            horas_semanales_requeridas=20,
            disponibilidad=disponibilidad_completa,
            materias=["Matemática", "Física"]
        ),
        Docente(
            id=1, 
            nombre="Carlos Mendoza", 
            tipo="contratado", 
            horas_semanales_requeridas=15,
            disponibilidad=disponibilidad_parcial,
            materias=["Computación", "Tecnología"]
        ),
        Docente(
            id=2, 
            nombre="María López", 
            tipo="nombrado", 
            horas_semanales_requeridas=18,
            disponibilidad=disponibilidad_completa,
            materias=["Comunicación", "Historia", "Arte"]
        ),
        Docente(
            id=3, 
            nombre="Ana Torres", 
            tipo="contratado", 
            horas_semanales_requeridas=12,
            disponibilidad=disponibilidad_completa,
            materias=["Inglés"]
        ),
    ]
    
    # Crear grados
    grados = [
        Grado(id=0, numero=3, num_secciones=2),  # 3A, 3B
        Grado(id=1, numero=4, num_secciones=2),  # 4A, 4B
        Grado(id=2, numero=5, num_secciones=1),  # 5A
    ]
    
    # Crear bloques de clases
    bloques = [
        # === GRADO 3A (grado_id=0, seccion_idx=0) ===
        BloqueClase(0, docente_id=0, grado_id=0, seccion_idx=0, 
                   materia="Matemática", tipo_bloque=TipoBloque.DOBLE),
        BloqueClase(1, docente_id=0, grado_id=0, seccion_idx=0, 
                   materia="Matemática", tipo_bloque=TipoBloque.DOBLE),
        BloqueClase(2, docente_id=1, grado_id=0, seccion_idx=0, 
                   materia="Computación", tipo_bloque=TipoBloque.TRIPLE),
        BloqueClase(3, docente_id=2, grado_id=0, seccion_idx=0, 
                   materia="Comunicación", tipo_bloque=TipoBloque.DOBLE),
        BloqueClase(4, docente_id=3, grado_id=0, seccion_idx=0, 
                   materia="Inglés", tipo_bloque=TipoBloque.SIMPLE),
        
        # === GRADO 3B (grado_id=0, seccion_idx=1) ===
        BloqueClase(5, docente_id=0, grado_id=0, seccion_idx=1, 
                   materia="Matemática", tipo_bloque=TipoBloque.DOBLE),
        BloqueClase(6, docente_id=2, grado_id=0, seccion_idx=1, 
                   materia="Historia", tipo_bloque=TipoBloque.DOBLE),
        BloqueClase(7, docente_id=2, grado_id=0, seccion_idx=1, 
                   materia="Arte", tipo_bloque=TipoBloque.SIMPLE),
        BloqueClase(8, docente_id=3, grado_id=0, seccion_idx=1, 
                   materia="Inglés", tipo_bloque=TipoBloque.DOBLE),
        
        # === GRADO 4A (grado_id=1, seccion_idx=0) ===
        BloqueClase(9, docente_id=0, grado_id=1, seccion_idx=0, 
                   materia="Física", tipo_bloque=TipoBloque.TRIPLE),
        BloqueClase(10, docente_id=0, grado_id=1, seccion_idx=0, 
                   materia="Matemática", tipo_bloque=TipoBloque.DOBLE),
        BloqueClase(11, docente_id=1, grado_id=1, seccion_idx=0, 
                   materia="Computación", tipo_bloque=TipoBloque.DOBLE),
        BloqueClase(12, docente_id=2, grado_id=1, seccion_idx=0, 
                   materia="Comunicación", tipo_bloque=TipoBloque.DOBLE),
        BloqueClase(13, docente_id=2, grado_id=1, seccion_idx=0, 
                   materia="Historia", tipo_bloque=TipoBloque.DOBLE),
        
        # === GRADO 4B (grado_id=1, seccion_idx=1) ===
        BloqueClase(14, docente_id=0, grado_id=1, seccion_idx=1, 
                   materia="Matemática", tipo_bloque=TipoBloque.TRIPLE),
        BloqueClase(15, docente_id=1, grado_id=1, seccion_idx=1, 
                   materia="Tecnología", tipo_bloque=TipoBloque.DOBLE),
        BloqueClase(16, docente_id=2, grado_id=1, seccion_idx=1, 
                   materia="Comunicación", tipo_bloque=TipoBloque.DOBLE),
        BloqueClase(17, docente_id=3, grado_id=1, seccion_idx=1, 
                   materia="Inglés", tipo_bloque=TipoBloque.DOBLE),
        
        # === GRADO 5A (grado_id=2, seccion_idx=0) ===
        BloqueClase(18, docente_id=0, grado_id=2, seccion_idx=0, 
                   materia="Física", tipo_bloque=TipoBloque.DOBLE),
        BloqueClase(19, docente_id=0, grado_id=2, seccion_idx=0, 
                   materia="Matemática", tipo_bloque=TipoBloque.TRIPLE),
        BloqueClase(20, docente_id=1, grado_id=2, seccion_idx=0, 
                   materia="Computación", tipo_bloque=TipoBloque.TRIPLE),
        BloqueClase(21, docente_id=2, grado_id=2, seccion_idx=0, 
                   materia="Historia", tipo_bloque=TipoBloque.DOBLE),
        BloqueClase(22, docente_id=3, grado_id=2, seccion_idx=0, 
                   materia="Inglés", tipo_bloque=TipoBloque.TRIPLE),
    ]
    
    # Crear y ejecutar el optimizador
    scheduler = AlgoritmoGeneticoHorarios(
        docentes=docentes,
        grados=grados,
        bloques=bloques,
        num_generaciones=200,
        tamano_poblacion=150
    )
    
    # Ejecutar optimización
    mejor_solucion, ga_instance = scheduler.ejecutar()
    
    # Mostrar resultados
    scheduler.imprimir_horarios(mejor_solucion)
    
    # Mostrar estadísticas de docentes
    print("\n" + "="*100)
    print("📊 ESTADÍSTICAS DE DOCENTES")
    print("="*100)
    stats_docentes = scheduler.generar_estadisticas_docentes(mejor_solucion)
    print(stats_docentes.to_string(index=False))
    
    # Exportar a Excel (opcional)
    # scheduler.exportar_horarios(mejor_solucion, "horarios_optimizados")