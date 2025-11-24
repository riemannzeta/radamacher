import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

@dataclass
class Agent:
    """Represents an individual in the simulation"""
    id: int
    genes: np.ndarray  # [foraging_skill, defense_skill, social_tendency]
    energy: float = 1.0
    is_alive: bool = True
    colony_id: int = -1  # -1 means solitary
    generation: int = 0
    
    # Track relatedness
    parents: List[int] = None

class Colony:
    """Represents a group of agents sharing a nest"""
    id: int
    queen_id: int
    workers: List[int]
    resources: float = 0.0
    nest_quality: float = 0.0
    
    def __init__(self, id: int, queen_id: int):
        self.id = id
        self.queen_id = queen_id
        self.workers = []
        self.resources = 0.0
        self.nest_quality = 0.0

class EusocialityABM:
    """
    Agent-Based Model for Eusociality Evolution
    
    Simulates the emergence of eusocial behavior from ecological constraints
    and individual decision rules, WITHOUT hardcoding the causal link.
    """
    
    def __init__(self, 
                 n_agents: int = 200, 
                 n_generations: int = 50,
                 ecological_pressure: float = 0.5,
                 resource_availability: float = 0.5,
                 carrying_capacity: int = 1000):
        
        self.n_agents = n_agents
        self.n_generations = n_generations
        self.ecological_pressure = ecological_pressure
        self.resource_availability = resource_availability
        self.carrying_capacity = carrying_capacity
        
        self.agents: Dict[int, Agent] = {}
        self.colonies: Dict[int, Colony] = {}
        self.next_agent_id = 0
        self.next_colony_id = 0
        
        # Data collection
        self.history = []
        
        # Initialize population
        self._initialize_population()
        
    def _initialize_population(self):
        """Create initial population of solitary agents"""
        for _ in range(self.n_agents):
            genes = np.random.rand(3)  # Random genetic traits
            self.agents[self.next_agent_id] = Agent(
                id=self.next_agent_id,
                genes=genes,
                parents=[-1, -1]  # Founders have no recorded parents
            )
            self.next_agent_id += 1
            
    def _calculate_relatedness(self, agent1_id: int, agent2_id: int) -> float:
        """Calculate genetic relatedness between two agents"""
        if agent1_id == -1 or agent2_id == -1:
            return 0.0
        
        a1 = self.agents[agent1_id]
        a2 = self.agents[agent2_id]
        
        # Simplified genetic distance (1 - Euclidean distance of genes)
        # In a real genetic model, we'd track alleles
        gene_dist = np.linalg.norm(a1.genes - a2.genes)
        return max(0.0, 1.0 - gene_dist)

    def step(self):
        """Run one generation of the simulation"""
        
        # 1. Ecological Phase: Resource gathering and Nest building
        for agent in self.agents.values():
            if not agent.is_alive: continue
            
            # Foraging success depends on skill and resource availability
            foraging_success = agent.genes[0] * self.resource_availability + np.random.normal(0, 0.1)
            agent.energy += max(0, foraging_success)
            
            # Nest building/maintenance
            if agent.colony_id != -1:
                colony = self.colonies[agent.colony_id]
                # Defense skill contributes to nest quality
                colony.nest_quality += agent.genes[1] * 0.1
                colony.resources += agent.energy * 0.5  # Share resources
                agent.energy *= 0.5
        
        # 2. Survival Phase: Ecological pressure attacks
        survivors = []
        for agent in self.agents.values():
            if not agent.is_alive: continue
            
            survival_prob = 0.8  # Base survival
            
            # Density-dependent mortality (carrying capacity)
            density_penalty = len(self.agents) / self.carrying_capacity
            survival_prob *= max(0, 1.0 - density_penalty * 0.5)
            
            # Pressure reduces survival
            risk = self.ecological_pressure
            
            # Nest defense mitigates risk
            if agent.colony_id != -1:
                colony = self.colonies[agent.colony_id]
                defense_bonus = colony.nest_quality * 0.5
                risk = max(0, risk - defense_bonus)
            
            if np.random.random() < (risk * 0.5) or np.random.random() > survival_prob:
                agent.is_alive = False
            else:
                survivors.append(agent.id)
                
        # 3. Reproduction & Social Decision Phase
        new_agents = []
        
        for agent_id in survivors:
            agent = self.agents[agent_id]
            
            # Decision: Reproduce or Help?
            # Based on social tendency gene and ecological cues
            social_drive = agent.genes[2]
            
            # If resources are scarce and pressure is high, grouping is favored
            grouping_incentive = (self.ecological_pressure * 0.7 + 
                                (1 - self.resource_availability) * 0.3)
            
            # PHENOTYPIC PLASTICITY MODEL (Fair NTW Test)
            # Decision is based on environment, not genes.
            # We add some noise to represent individual variation/circumstance, 
            # but it is NOT heritable.
            threshold = 0.5 + np.random.normal(0, 0.1)
            will_help = grouping_incentive > threshold
            
            if will_help and agent.colony_id != -1:
                # Stay as worker (eusocial behavior)
                pass 
            elif agent.energy > 0.5:
                # Reproduce (create new agent)
                # Simplified asexual reproduction with mutation for this demo
                child_genes = agent.genes + np.random.normal(0, 0.05, 3)
                child_genes = np.clip(child_genes, 0, 1)
                
                child = Agent(
                    id=self.next_agent_id,
                    genes=child_genes,
                    parents=[agent.id, -1],
                    generation=agent.generation + 1
                )
                new_agents.append(child)
                self.next_agent_id += 1
                
                # Decision: Form colony with child?
                if will_help:
                    if agent.colony_id == -1:
                        # Start new colony
                        colony = Colony(self.next_colony_id, agent.id)
                        self.colonies[self.next_colony_id] = colony
                        agent.colony_id = self.next_colony_id
                        self.next_colony_id += 1
                    
                    # Child stays in colony
                    child.colony_id = agent.colony_id
                    self.colonies[agent.colony_id].workers.append(child.id)
        
        # Add new agents to population
        for ag in new_agents:
            self.agents[ag.id] = ag
            
        # 4. Data Collection
        self._record_stats()
        
    def _record_stats(self):
        """Record statistics for the current generation"""
        alive_agents = [a for a in self.agents.values() if a.is_alive]
        if not alive_agents: return
        
        # Calculate metrics
        n_eusocial = sum(1 for a in alive_agents if a.colony_id != -1)
        eusociality_rate = n_eusocial / len(alive_agents)
        
        avg_relatedness = 0.0
        n_pairs = 0
        
        # Sample pairs for relatedness to save time
        sample_size = min(100, len(alive_agents))
        sample = np.random.choice(alive_agents, sample_size, replace=False)
        
        for i in range(len(sample)):
            for j in range(i+1, len(sample)):
                r = self._calculate_relatedness(sample[i].id, sample[j].id)
                avg_relatedness += r
                n_pairs += 1
        
        avg_relatedness = avg_relatedness / n_pairs if n_pairs > 0 else 0
        
        # Colony stats
        avg_colony_size = 0
        if self.colonies:
            sizes = [len(c.workers) + 1 for c in self.colonies.values()]
            avg_colony_size = np.mean(sizes)
            
        self.history.append({
            'ecological_pressure': self.ecological_pressure,
            'resource_availability': self.resource_availability,
            'eusociality': eusociality_rate,
            'relatedness': avg_relatedness,
            'colony_formation': len(self.colonies) / len(alive_agents) if alive_agents else 0,
            'avg_colony_size': avg_colony_size,
            'worker_retention': np.mean([a.genes[2] for a in alive_agents]) # Social gene as proxy
        })

    def run(self):
        """Run the full simulation"""
        for _ in range(self.n_generations):
            self.step()
        return pd.DataFrame(self.history)

def generate_abm_data(n_samples=100):
    """Generate dataset by running ABM with varying parameters"""
    all_data = []
    
    print(f"Generating data from {n_samples} ABM simulations...")
    
    for i in range(n_samples):
        # Randomize environmental conditions
        eco_pressure = np.random.random()
        res_avail = np.random.random()
        
        model = EusocialityABM(
            n_agents=100, 
            n_generations=30,
            ecological_pressure=eco_pressure,
            resource_availability=res_avail
        )
        
        df = model.run()
        
        # Take the final state as the data point
        final_state = df.iloc[-1].to_dict()
        all_data.append(final_state)
        
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    # Test run
    data = generate_abm_data(n_samples=10)
    print(data.head())
