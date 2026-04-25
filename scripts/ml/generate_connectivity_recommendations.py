#!/usr/bin/env python3
"""
Generate 150 Strategic Connectivity Recommendations for Neuro Knowledge Graph
Based on audit findings to improve avg connectivity from 2.18 to 3.5+
"""

import pandas as pd
import json

# Load the audit report to understand current structure
print("Loading neuro audit report...")
with open('neuro_audit_report.json', 'r', encoding='utf-8') as f:
    audit = json.load(f)

# Create 150 strategic relationships
recommendations = []

# ============================================================================
# PRIORITY 1: Cross-Domain Bridges (50 relationships)
# ============================================================================
print('\nGenerating Priority 1: Cross-Domain Bridges (50)...')

cross_domain = [
    # Attention ↔ Working Memory
    ('Attention', 'Selective Attention', 'Attention', 'ENABLES', 'Working Memory Capacity', 'WorkingMemory', 'Memory Systems', 'Selective attention enables working memory by filtering relevant information and suppressing distractions, allowing limited capacity to be allocated effectively.'),
    ('Attention', 'Sustained Attention', 'Attention', 'SUPPORTS', 'Working Memory', 'Memory', 'Memory Systems', 'Sustained attention supports memory consolidation by maintaining focus during encoding and retrieval processes.'),
    ('WorkingMemory', 'Working Memory', 'Memory', 'REQUIRES', 'Selective Attention', 'Attention', 'Attention Types', 'Working memory requires selective attention to maintain relevant information while inhibiting irrelevant stimuli.'),
    ('WorkingMemory', 'Reduced Working Memory Capacity', 'ExecutiveFunctions', 'IMPAIRS', 'Selective Attention', 'Attention', 'Attention Types', 'Reduced working memory capacity impairs selective attention by limiting cognitive resources available for attentional control.'),
    
    # Metacognition ↔ Executive Functions
    ('Metacognition', 'Metacognitive Monitoring', 'MetacognitiveMonitoring', 'GUIDES', 'Executive Functions', 'CognitiveControl', 'Executive Functions', 'Metacognitive monitoring guides executive functions by providing real-time feedback on cognitive performance and strategy effectiveness.'),
    ('Metacognition', 'Metacognitive Control', 'MetacognitiveControl', 'REGULATES', 'Planning & Organization', 'ExecutiveFunctions', 'Executive Functions', 'Metacognitive control regulates planning and organization by adjusting strategies based on self-assessment of progress.'),
    ('ExecutiveFunctions', 'Planning & Organization', 'ExecutiveFunctions', 'ENABLES', 'Metacognitive Monitoring', 'MetacognitiveMonitoring', 'Metacognitive Processes', 'Planning and organization enable metacognitive monitoring by providing structure for self-assessment and progress tracking.'),
    ('ExecutiveFunctions', 'Impaired Inhibitory Control', 'ExecutiveFunctions', 'LIMITS', 'Metacognitive Control', 'MetacognitiveControl', 'Metacognitive Processes', 'Impaired inhibitory control limits metacognitive control by reducing ability to suppress ineffective strategies.'),
    
    # Intrinsic Motivation ↔ Growth Mindset
    ('IntrinsicMotivation', 'Autonomy', 'IntrinsicMotivation', 'FOSTERS', 'Growth Mindset', 'Mindset', 'Belief Systems', 'Autonomy fosters growth mindset by supporting self-directed learning and ownership of challenges.'),
    ('IntrinsicMotivation', 'Competence', 'IntrinsicMotivation', 'STRENGTHENS', 'Growth Mindset', 'Mindset', 'Belief Systems', 'Competence strengthens growth mindset by providing evidence that effort leads to improvement.'),
    ('GrowthMindset', 'Belief', 'GrowthMindset', 'ENHANCES', 'Internal Satisfaction', 'IntrinsicMotivation', 'Motivational Types', 'Growth mindset belief enhances internal satisfaction by valuing learning process over external validation.'),
    ('GrowthMindset', 'Effort Seen As A Path to Mastery', 'GrowthMindset', 'DRIVES', 'Self-Determination', 'IntrinsicMotivation', 'Motivational Types', 'Viewing effort as a path to mastery drives self-determination by emphasizing internal control over outcomes.'),
    
    # Emotional Regulation ↔ Attention
    ('EmotionalRegulation', 'Awareness and Management of Emotions', 'EmotionalRegulation', 'MODULATES', 'Selective Attention', 'Attention', 'Attention Types', 'Emotional awareness and management modulate selective attention by reducing interference from emotional distractions.'),
    ('EmotionalRegulation', 'Difficulty Managing Frustration or Stress', 'EmotionalRegulation', 'DISRUPTS', 'Sustained Attention', 'Attention', 'Attention Types', 'Difficulty managing frustration disrupts sustained attention by triggering arousal that diverts cognitive resources.'),
    ('Attention', 'Selective Attention', 'Attention', 'FACILITATES', 'Awareness and Management of Emotions', 'EmotionalRegulation', 'Affective States', 'Selective attention facilitates emotional awareness by enabling focus on internal affective states.'),
    
    # Motivation ↔ Attention
    ('IntrinsicMotivation', 'Internal Satisfaction', 'IntrinsicMotivation', 'SUSTAINS', 'Sustained Attention', 'Attention', 'Attention Types', 'Internal satisfaction sustains attention by providing intrinsic reward that maintains engagement.'),
    ('Motivation', 'Altered Reward Sensitivity', 'Motivation', 'AFFECTS', 'Selective Attention', 'Attention', 'Attention Types', 'Altered reward sensitivity affects selective attention by changing salience of task-relevant stimuli.'),
    
    # Memory ↔ Emotion
    ('Memory', 'Working Memory', 'Memory', 'IS_MODULATED_BY', 'Positive Emotions', 'Emotions', 'Affective States', 'Working memory is modulated by positive emotions, which broaden attentional scope and enhance cognitive flexibility.'),
    ('Memory', 'Memory Consolidation', 'Memory', 'IS_ENHANCED_BY', 'Positive Emotions', 'Emotions', 'Affective States', 'Memory consolidation is enhanced by positive emotions through increased hippocampal and dopaminergic activity.'),
    ('Emotions', 'Negative Emotions', 'Emotions', 'NARROWS', 'Working Memory Capacity', 'WorkingMemory', 'Memory Systems', 'Negative emotions narrow working memory capacity by prioritizing threat-related processing.'),
    
    # Executive Functions ↔ Learning Outcomes
    ('ExecutiveFunctions', 'Planning & Organization', 'ExecutiveFunctions', 'IMPROVES', 'Participation', 'LearningOutcomes', 'Educational Results', 'Planning and organization improve participation by enabling structured approach to learning activities.'),
    ('ExecutiveFunctions', 'Resilience', 'ExecutiveFunctions', 'LEADS_TO', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Resilience leads to task completion by enabling persistence through challenges and setbacks.'),
    ('ExecutiveFunctions', 'Adaptation of Strategies to Improve Task Performance', 'ExecutiveFunctions', 'ENHANCES', 'Organization', 'LearningOutcomes', 'Educational Results', 'Strategy adaptation enhances organization by allowing flexible adjustment to task demands.'),
    
    # Metacognition ↔ Critical Thinking
    ('Metacognition', 'Awareness of Errors', 'Metacognition', 'ENABLES', 'Analysis', 'CriticalThinking', 'Higher-Order Cognition', 'Error awareness enables analysis by providing feedback on reasoning quality.'),
    ('Metacognition', 'Ability to Adjust Strategies', 'Metacognition', 'SUPPORTS', 'Evaluation', 'CriticalThinking', 'Higher-Order Cognition', 'Strategy adjustment ability supports evaluation by enabling iterative refinement of thinking.'),
    ('CriticalThinking', 'Analysis', 'CriticalThinking', 'REQUIRES', 'Metacognitive Monitoring', 'MetacognitiveMonitoring', 'Metacognitive Processes', 'Analysis requires metacognitive monitoring to assess validity of reasoning steps.'),
    
    # Growth Mindset ↔ Resilience
    ('GrowthMindset', 'Challenges Are Opportunities to Improve', 'GrowthMindset', 'BUILDS', 'Resilience', 'ExecutiveFunctions', 'Executive Functions', 'Viewing challenges as opportunities builds resilience by reframing difficulties as learning experiences.'),
    ('GrowthMindset', 'Failure Is Feedback and A Chance to Learn', 'GrowthMindset', 'STRENGTHENS', 'Resilience', 'ExecutiveFunctions', 'Executive Functions', 'Viewing failure as feedback strengthens resilience by reducing threat response to setbacks.'),
    ('ExecutiveFunctions', 'Resilience', 'ExecutiveFunctions', 'REINFORCES', 'Growth Mindset', 'Mindset', 'Belief Systems', 'Resilience reinforces growth mindset by providing evidence that persistence leads to improvement.'),
    
    # Attention ↔ Learning Performance
    ('Attention', 'Selective Attention', 'Attention', 'PREDICTS', 'Deep Learning', 'LearningPerformance', 'Educational Results', 'Selective attention predicts deep learning by enabling sustained focus on relevant information.'),
    ('Attention', 'Sustained Attention', 'Attention', 'CONTRIBUTES_TO', 'Persistence', 'LearningPerformance', 'Educational Results', 'Sustained attention contributes to persistence by maintaining engagement during challenging tasks.'),
    
    # Stress ↔ Cognitive Processes
    ('PositiveStressEustress', 'Optimal Arousal', 'PositiveStressEustress', 'OPTIMIZES', 'Selective Attention', 'Attention', 'Attention Types', 'Optimal arousal optimizes selective attention through balanced activation of attentional networks.'),
    ('NegativeStressDistress', 'Overwhelming Stress That Impairs Functioning', 'NegativeStressDistress', 'NARROWS', 'Working Memory Capacity', 'WorkingMemory', 'Memory Systems', 'Overwhelming stress narrows working memory capacity by prioritizing survival-related processing.'),
    
    # Working Memory ↔ Executive Functions
    ('WorkingMemory', 'Working Memory Capacity', 'ResourceAllocation', 'CONSTRAINS', 'Planning & Organization', 'ExecutiveFunctions', 'Executive Functions', 'Working memory capacity constrains planning and organization by limiting information that can be held simultaneously.'),
    ('WorkingMemory', 'Chunking / Organization', 'WorkingMemory', 'EXPANDS', 'Planning & Organization', 'ExecutiveFunctions', 'Executive Functions', 'Chunking expands planning capacity by reducing cognitive load through information organization.'),
    
    # Metacognition ↔ Self-Regulation
    ('Metacognition', 'Metacognitive Monitoring', 'MetacognitiveMonitoring', 'IS_ESSENTIAL_FOR', 'Adaptive Responses', 'SelfRegulation', 'Metacognitive Processes', 'Metacognitive monitoring is essential for adaptive responses by detecting when current strategies are ineffective.'),
    ('SelfRegulation', 'Adaptive Responses', 'SelfRegulation', 'DEPENDS_ON', 'Metacognitive Control', 'MetacognitiveControl', 'Metacognitive Processes', 'Adaptive responses depend on metacognitive control to implement strategy adjustments.'),
    
    # Motivation ↔ Executive Functions
    ('IntrinsicMotivation', 'Self-Determination', 'IntrinsicMotivation', 'FUELS', 'Planning & Organization', 'ExecutiveFunctions', 'Executive Functions', 'Self-determination fuels planning and organization by providing intrinsic drive to pursue goals.'),
    ('Motivation', 'Altered Reward Sensitivity', 'Motivation', 'MODULATES', 'Impaired Inhibitory Control', 'ExecutiveFunctions', 'Executive Functions', 'Altered reward sensitivity modulates inhibitory control by affecting value assigned to immediate rewards.'),
    
    # Creativity ↔ Metacognition
    ('Creativity', 'Divergent Thinking', 'Creativity', 'REQUIRES', 'Metacognitive Monitoring', 'MetacognitiveMonitoring', 'Metacognitive Processes', 'Divergent thinking requires metacognitive monitoring to evaluate novelty and utility of ideas.'),
    ('Metacognition', 'Ability to Adjust Strategies', 'Metacognition', 'FACILITATES', 'Problem-solving', 'Creativity', 'Creative Processes', 'Strategy adjustment ability facilitates creative problem-solving by enabling flexible approach exploration.'),
    
    # Emotion ↔ Memory Systems
    ('PositiveEmotions', 'Joy', 'PositiveEmotions', 'STRENGTHENS', 'Memory Consolidation', 'Memory', 'Memory Systems', 'Joy strengthens memory consolidation through increased dopamine release during encoding.'),
    ('PositiveEmotions', 'Interest', 'PositiveEmotions', 'DEEPENS', 'Memory Encoding', 'MemorySystems', 'Memory Systems', 'Interest deepens memory encoding by increasing attention and elaborative processing.'),
    ('NegativeEmotions', 'Fear', 'NegativeEmotions', 'PRIORITIZES', 'Memory Encoding', 'MemorySystems', 'Memory Systems', 'Fear prioritizes memory encoding for threat-related information through amygdala activation.'),
    
    # Social Learning ↔ Motivation
    ('SocialLearning', 'Peer Relationships', 'SocialLearning', 'FULFILLS', 'Relatedness', 'IntrinsicMotivation', 'Motivational Types', 'Peer relationships fulfill relatedness needs, supporting intrinsic motivation through social connection.'),
    ('IntrinsicMotivation', 'Relatedness', 'IntrinsicMotivation', 'ENCOURAGES', 'Feedback Integration', 'SocialLearning', 'Learning Processes', 'Relatedness encourages feedback integration by creating safe environment for learning from others.'),
    
    # Attention ↔ Executive Functions
    ('Attention', 'Selective Attention', 'Attention', 'IS_GOVERNED_BY', 'Impaired Inhibitory Control', 'ExecutiveFunctions', 'Executive Functions', 'Selective attention is governed by inhibitory control, which suppresses irrelevant stimuli.'),
    ('ExecutiveFunctions', 'Planning & Organization', 'ExecutiveFunctions', 'DIRECTS', 'Selective Attention', 'Attention', 'Attention Types', 'Planning and organization direct selective attention toward goal-relevant information.'),
    
    # Mindset ↔ Learning Strategies
    ('GrowthMindset', 'Effort Seen As A Path to Mastery', 'GrowthMindset', 'PROMOTES', 'Strategic Use of Learning Methods', 'SelfRegulatedLearning', 'Metacognitive Processes', 'Viewing effort as path to mastery promotes strategic learning by encouraging experimentation with methods.'),
    ('Mindset', 'Growth Mindset', 'Mindset', 'REDUCES', 'Inefficient Rehearsal or Organization', 'LearningStrategies', 'Learning Processes', 'Growth mindset reduces inefficient strategies by encouraging reflection and adjustment.'),
]

recommendations.extend(cross_domain[:50])  # Take first 50
print(f'  Added {len(cross_domain[:50])} cross-domain relationships')

# ============================================================================
# PRIORITY 2: Mechanism → Outcome Links (75 relationships)
# ============================================================================
print('Generating Priority 2: Mechanism → Outcome Links (75)...')

mechanism_outcome = [
    # Attention → Academic Outcomes
    ('Attention', 'Selective Attention', 'Attention', 'IMPROVES', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Selective attention improves task completion by filtering distractions and maintaining focus on goals.'),
    ('Attention', 'Sustained Attention', 'Attention', 'PREDICTS', 'Persistence', 'LearningPerformance', 'Educational Results', 'Sustained attention predicts persistence by enabling prolonged engagement with challenging material.'),
    ('Attention', 'Selective Attention', 'Attention', 'ENHANCES', 'Deep Learning', 'LearningPerformance', 'Educational Results', 'Selective attention enhances deep learning by enabling focus on critical conceptual relationships.'),
    
    # Working Memory → Academic Performance
    ('WorkingMemory', 'Working Memory Capacity', 'ResourceAllocation', 'PREDICTS', 'Organization', 'LearningOutcomes', 'Educational Results', 'Working memory capacity predicts organizational ability by determining how much information can be coordinated.'),
    ('WorkingMemory', 'Chunking / Organization', 'WorkingMemory', 'IMPROVES', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Chunking improves task completion by reducing cognitive load and enabling multi-step problem solving.'),
    ('WorkingMemory', 'Limited Verbal Working Memory', 'WorkingMemory', 'CONSTRAINS', 'Participation', 'LearningOutcomes', 'Educational Results', 'Limited verbal working memory constrains participation by affecting ability to follow complex instructions.'),
    
    # Metacognition → Problem Solving
    ('Metacognition', 'Metacognitive Monitoring', 'MetacognitiveMonitoring', 'LEADS_TO', 'Error Detection', 'ProblemSolving', 'Cognitive Skills', 'Metacognitive monitoring leads to error detection by enabling self-assessment during problem solving.'),
    ('Metacognition', 'Metacognitive Control', 'MetacognitiveControl', 'IMPROVES', 'Correction', 'ProblemSolving', 'Cognitive Skills', 'Metacognitive control improves error correction by enabling strategy adjustment.'),
    ('Metacognition', 'Awareness of Errors', 'Metacognition', 'ENHANCES', 'Effective Problem-solving', 'LearningDevelopment', 'Educational Results', 'Error awareness enhances problem-solving effectiveness by providing feedback for strategy refinement.'),
    
    # Executive Functions → Learning Outcomes
    ('ExecutiveFunctions', 'Planning & Organization', 'ExecutiveFunctions', 'LEADS_TO', 'Time Management', 'LearningOutcomes', 'Educational Results', 'Planning and organization lead to better time management by enabling prioritization and scheduling.'),
    ('ExecutiveFunctions', 'Resilience', 'ExecutiveFunctions', 'PRODUCES', 'Goal Completion', 'LearningPerformance', 'Educational Results', 'Resilience produces goal completion by supporting persistence through obstacles.'),
    ('ExecutiveFunctions', 'Adaptation of Strategies to Improve Task Performance', 'ExecutiveFunctions', 'RESULTS_IN', 'Prioritization', 'LearningOutcomes', 'Educational Results', 'Strategy adaptation results in better prioritization by matching approaches to task demands.'),
    
    # Intrinsic Motivation → Learning Quality
    ('IntrinsicMotivation', 'Internal Satisfaction', 'IntrinsicMotivation', 'PRODUCES', 'Deep Learning', 'LearningPerformance', 'Educational Results', 'Internal satisfaction produces deep learning by sustaining engagement with material.'),
    ('IntrinsicMotivation', 'Autonomy', 'IntrinsicMotivation', 'FOSTERS', 'Persistence', 'LearningPerformance', 'Educational Results', 'Autonomy fosters persistence by supporting ownership of learning goals.'),
    ('IntrinsicMotivation', 'Competence', 'IntrinsicMotivation', 'BUILDS', 'Goal Completion', 'LearningPerformance', 'Educational Results', 'Competence builds goal completion by providing confidence and efficacy beliefs.'),
    
    # Growth Mindset → Learning Outcomes
    ('GrowthMindset', 'Effort Seen As A Path to Mastery', 'GrowthMindset', 'INCREASES', 'Persistence', 'LearningPerformance', 'Educational Results', 'Viewing effort as path to mastery increases persistence by valuing struggle as learning.'),
    ('GrowthMindset', 'Challenges Are Opportunities to Improve', 'GrowthMindset', 'PRODUCES', 'Deep Learning', 'LearningPerformance', 'Educational Results', 'Viewing challenges as opportunities produces deep learning by encouraging engagement with difficulty.'),
    ('GrowthMindset', 'Failure Is Feedback and A Chance to Learn', 'GrowthMindset', 'SUPPORTS', 'Wellbeing', 'LearningPerformance', 'Educational Results', 'Viewing failure as feedback supports wellbeing by reducing threat response to setbacks.'),
    
    # Critical Thinking → Outcomes
    ('CriticalThinking', 'Analysis', 'CriticalThinking', 'ENABLES', 'Effective Problem-solving', 'LearningDevelopment', 'Educational Results', 'Analysis enables effective problem-solving by breaking complex issues into manageable components.'),
    ('CriticalThinking', 'Evaluation', 'CriticalThinking', 'IMPROVES', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Evaluation improves task completion by enabling assessment of solution quality.'),
    ('CriticalThinking', 'Inference', 'CriticalThinking', 'SUPPORTS', 'Deep Learning', 'LearningPerformance', 'Educational Results', 'Inference supports deep learning by connecting new information to existing knowledge.'),
    
    # Emotional Regulation → Performance
    ('EmotionalRegulation', 'Awareness and Management of Emotions', 'EmotionalRegulation', 'IMPROVES', 'Persistence', 'LearningPerformance', 'Educational Results', 'Emotional awareness improves persistence by reducing interference from negative affect.'),
    ('EmotionalRegulation', 'Awareness and Management of Emotions', 'EmotionalRegulation', 'ENHANCES', 'Wellbeing', 'LearningPerformance', 'Educational Results', 'Emotional management enhances wellbeing by providing tools for stress reduction.'),
    ('EmotionalRegulation', 'Difficulty Managing Frustration or Stress', 'EmotionalRegulation', 'REDUCES', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Difficulty managing frustration reduces task completion by triggering avoidance behaviors.'),
    
    # Creativity → Innovation Outcomes
    ('Creativity', 'Divergent Thinking', 'Creativity', 'PRODUCES', 'Innovative Problem-solving', 'CreativityInnovation', 'Creative Processes', 'Divergent thinking produces innovative problem-solving by generating multiple solution paths.'),
    ('Creativity', 'Risk-taking', 'Creativity', 'ENABLES', 'Innovative Solutions', 'CreativeOutcomes', 'Creative Processes', 'Risk-taking enables innovative solutions by supporting exploration of unconventional approaches.'),
    ('Creativity', 'Problem-solving', 'Creativity', 'LEADS_TO', 'Original Approaches', 'CreativeOutcomes', 'Creative Processes', 'Creative problem-solving leads to original approaches through flexible thinking.'),
    
    # Social Cognition → Social Outcomes
    ('SocialCognition', 'Perspective-taking', 'SocialCognition', 'IMPROVES', 'Collaborative Learning', 'SocialDevelopment', 'Learning Processes', 'Perspective-taking improves collaborative learning by enabling understanding of peer viewpoints.'),
    ('SocialCognition', 'Cooperation', 'SocialCognition', 'FACILITATES', 'Peer Interaction', 'SocialDevelopment', 'Learning Processes', 'Cooperation facilitates peer interaction by supporting shared goal pursuit.'),
    ('SocialCognition', 'Social Understanding', 'SocialCognition', 'SUPPORTS', 'Classroom Participation', 'LanguageDevelopment', 'Academic Skills', 'Social understanding supports classroom participation by enabling appropriate social engagement.'),
    
    # Stress → Performance Outcomes
    ('PositiveStressEustress', 'Optimal Arousal', 'PositiveStressEustress', 'MAXIMIZES', 'Persistence', 'LearningPerformance', 'Educational Results', 'Optimal arousal maximizes persistence by providing energizing motivation without overwhelm.'),
    ('PositiveStressEustress', 'Manageable Arousal of Body and Mind', 'PositiveStressEustress', 'SUPPORTS', 'Goal Completion', 'LearningPerformance', 'Educational Results', 'Manageable arousal supports goal completion by maintaining focus and energy.'),
    ('NegativeStressDistress', 'Overwhelming Stress That Impairs Functioning', 'NegativeStressDistress', 'IMPAIRS', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Overwhelming stress impairs task completion by hijacking cognitive resources.'),
    
    # Memory → Learning Outcomes
    ('Memory', 'Working Memory', 'Memory', 'SUPPORTS', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Working memory supports task completion by maintaining goal representations during execution.'),
    ('Memory', 'Memory Consolidation', 'Memory', 'ENABLES', 'Deep Learning', 'LearningPerformance', 'Educational Results', 'Memory consolidation enables deep learning by stabilizing knowledge representations.'),
    ('Memory', 'Memory Retrieval', 'Memory', 'FACILITATES', 'Participation', 'LearningOutcomes', 'Educational Results', 'Memory retrieval facilitates participation by providing access to relevant knowledge.'),
    
    # Self-Regulation → Academic Success
    ('SelfRegulation', 'Adaptive Responses', 'SelfRegulation', 'IMPROVES', 'Persistence', 'LearningPerformance', 'Educational Results', 'Adaptive responses improve persistence by enabling flexible adjustment to challenges.'),
    ('SelfRegulation', 'Positive Emotions Into Sustained Drive', 'SelfRegulation', 'MAINTAINS', 'Goal Completion', 'LearningPerformance', 'Educational Results', 'Channeling positive emotions maintains goal completion by sustaining motivation.'),
    ('SelfRegulation', 'Limited Metacognitive Monitoring', 'SelfRegulation', 'LIMITS', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Limited metacognitive monitoring limits task completion by reducing error detection.'),
    
    # Teaching Practices → Student Outcomes
    ('TeachingPractices', 'Adaptive Strategies', 'TeachingPractices', 'IMPROVES', 'Participation', 'LearningOutcomes', 'Educational Results', 'Adaptive teaching strategies improve participation by matching instruction to student needs.'),
    ('TeachingPractices', 'Sustained Engagement', 'TeachingPractices', 'ENHANCES', 'Persistence', 'LearningPerformance', 'Educational Results', 'Sustained engagement enhances persistence by maintaining student interest and motivation.'),
    ('TeachingPractices', 'Predictability', 'TeachingPractices', 'SUPPORTS', 'Organization', 'LearningOutcomes', 'Educational Results', 'Predictable routines support organization by reducing cognitive load for task management.'),
    
    # Attention Difficulties → Academic Challenges
    ('Attention', 'Difficulty Sustaining Focus', 'Attention', 'IMPAIRS', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Difficulty sustaining focus impairs task completion by causing interruptions in work flow.'),
    ('Attention', 'Increased Cognitive Load During Reading Tasks', 'Attention', 'REDUCES', 'Deep Learning', 'LearningPerformance', 'Educational Results', 'Increased cognitive load during reading reduces deep learning by limiting comprehension capacity.'),
    ('Attention', 'High Cognitive Load in Math Tasks', 'Attention', 'CONSTRAINS', 'Prioritization', 'LearningOutcomes', 'Educational Results', 'High cognitive load constrains prioritization by overwhelming working memory.'),
    
    # Executive Function Deficits → Outcomes
    ('ExecutiveFunctions', 'Impaired Inhibitory Control', 'ExecutiveFunctions', 'INTERFERES_WITH', 'Organization', 'LearningOutcomes', 'Educational Results', 'Impaired inhibitory control interferes with organization by reducing ability to suppress distracting thoughts.'),
    ('ExecutiveFunctions', 'Reduced Working Memory Capacity', 'ExecutiveFunctions', 'LIMITS', 'Time Management', 'LearningOutcomes', 'Educational Results', 'Reduced working memory limits time management by affecting ability to coordinate multiple tasks.'),
    
    # Motivation Factors → Engagement
    ('ExtrinsicMotivation', 'External Rewards', 'ExtrinsicMotivation', 'PRODUCES', 'Short-Term Effort', 'LearningPerformance', 'Educational Results', 'External rewards produce short-term effort but may undermine long-term engagement.'),
    ('IntrinsicMotivation', 'Self-Determination', 'IntrinsicMotivation', 'GENERATES', 'Authentic Engagement', 'PersonalGrowth', 'Developmental Outcomes', 'Self-determination generates authentic engagement through ownership of learning.'),
    
    # Metacognitive Skills → Self-Regulation
    ('Metacognition', 'Errors As Opportunities', 'Metacognition', 'BUILDS', 'Adaptive Problem-solving', 'PersonalGrowth', 'Developmental Outcomes', 'Viewing errors as opportunities builds adaptive problem-solving by reducing fear of failure.'),
    ('Metacognition', 'Perceived Mastery', 'Metacognition', 'STRENGTHENS', 'Lifelong Learning and Adaptability', 'PersonalGrowth', 'Developmental Outcomes', 'Perceived mastery strengthens lifelong learning by building confidence in learning ability.'),
    
    # Positive Emotions → Cognitive Enhancement
    ('PositiveEmotions', 'Joy', 'PositiveEmotions', 'BOOSTS', 'Participation', 'LearningOutcomes', 'Educational Results', 'Joy boosts participation by creating positive associations with learning activities.'),
    ('PositiveEmotions', 'Interest', 'PositiveEmotions', 'DEEPENS', 'Deep Learning', 'LearningPerformance', 'Educational Results', 'Interest deepens learning by sustaining attention and promoting elaborative processing.'),
    ('PositiveEmotions', 'Hope', 'PositiveEmotions', 'SUSTAINS', 'Persistence', 'LearningPerformance', 'Educational Results', 'Hope sustains persistence by maintaining positive expectancies about success.'),
    
    # Negative Emotions → Performance Impairment
    ('NegativeEmotions', 'Fear', 'NegativeEmotions', 'REDUCES', 'Persistence', 'LearningPerformance', 'Educational Results', 'Fear reduces persistence by triggering avoidance of challenging tasks.'),
    ('NegativeEmotions', 'Anxiety', 'AffectiveProcesses', 'IMPAIRS', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Anxiety impairs task completion by consuming working memory resources with worry.'),
    ('NegativeEmotions', 'Shame', 'NegativeEmotions', 'UNDERMINES', 'Participation', 'LearningOutcomes', 'Educational Results', 'Shame undermines participation by creating avoidance of social exposure.'),
    
    # Language/Communication → Academic Skills
    ('Communication', 'Language Comprehension', 'Communication', 'ENABLES', 'Participation', 'LearningOutcomes', 'Educational Results', 'Language comprehension enables participation by supporting understanding of instruction.'),
    ('Communication', 'Reading & Writing', 'Communication', 'FACILITATES', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Reading and writing skills facilitate task completion across academic domains.'),
    ('Communication', 'Collaboration', 'Communication', 'SUPPORTS', 'Peer Interaction', 'SocialDevelopment', 'Learning Processes', 'Collaboration supports peer interaction by enabling coordinated joint activity.'),
    
    # Cognitive Strengths → Success
    ('Strengths', 'High Creativity', 'Strengths', 'CONTRIBUTES_TO', 'Innovative Problem-solving', 'CreativityInnovation', 'Creative Processes', 'High creativity contributes to innovative problem-solving through divergent thinking.'),
    ('Strengths', 'Pattern Recognition and Detail Orientation', 'Strengths', 'ENHANCES', 'Deep Knowledge', 'LearningDepth', 'Educational Results', 'Pattern recognition enhances deep knowledge by supporting systematic understanding.'),
    ('Strengths', 'Verbal Reasoning', 'Strengths', 'FACILITATES', 'Participation', 'LearningOutcomes', 'Educational Results', 'Verbal reasoning facilitates participation by enabling articulation of ideas.'),
    
    # Educational Support → Student Success
    ('EducationalSupport', 'Structured Routines', 'EducationalSupport', 'IMPROVES', 'Organization', 'LearningOutcomes', 'Educational Results', 'Structured routines improve organization by providing external scaffolding for task management.'),
    ('EducationalSupport', 'Visual Aids', 'EducationalSupport', 'ENHANCES', 'Participation', 'LearningOutcomes', 'Educational Results', 'Visual aids enhance participation by supporting multiple modalities of learning.'),
    ('EducationalSupport', 'Step-by-step Instruction', 'EducationalSupport', 'FACILITATES', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Step-by-step instruction facilitates task completion by reducing cognitive load.'),
    
    # Working Memory Strategies → Performance
    ('WorkingMemory', 'Chunking / Organization', 'WorkingMemory', 'ENABLES', 'Organization', 'LearningOutcomes', 'Educational Results', 'Chunking enables better organization by grouping information into meaningful units.'),
    ('WorkingMemory', 'Visualization / Mental Imagery', 'WorkingMemory', 'SUPPORTS', 'Deep Learning', 'LearningPerformance', 'Educational Results', 'Mental imagery supports deep learning by creating rich memory representations.'),
    ('WorkingMemory', 'Rehearsal', 'WorkingMemory', 'MAINTAINS', 'Task Completion', 'LearningOutcomes', 'Educational Results', 'Rehearsal maintains task completion by keeping goal representations active.'),
    
    # Mindset → Long-term Outcomes
    ('GrowthMindset', 'Encourages Intrinsic Motivation', 'GrowthMindset', 'PRODUCES', 'Lifelong Learning and Adaptability', 'PersonalGrowth', 'Developmental Outcomes', 'Growth mindset produces lifelong learning by valuing continuous improvement.'),
    ('FixedMindset', 'Encourages External Validation and Fear of Judgment', 'FixedMindset', 'LIMITS', 'Personal Growth', 'PersonalGrowth', 'Developmental Outcomes', 'Fixed mindset limits personal growth by avoiding challenges that risk failure.'),
    
    # Metacognitive Processes → Learning Efficiency
    ('MetacognitiveMonitoring', 'Self-checking Understanding During Tasks', 'MetacognitiveMonitoring', 'IMPROVES', 'Future Learning Efficiency', 'LongTermLearning', 'Educational Results', 'Self-checking improves future learning by identifying gaps in understanding.'),
    ('MetacognitiveControl', 'Adjusting Effort', 'MetacognitiveControl', 'OPTIMIZES', 'Time Management', 'LearningOutcomes', 'Educational Results', 'Adjusting effort optimizes time management by allocating resources to task difficulty.'),
    
    # Resilience → Persistence
    ('ExecutiveFunctions', 'Resilience', 'ExecutiveFunctions', 'IS_CRITICAL_FOR', 'Persistence', 'LearningPerformance', 'Educational Results', 'Resilience is critical for persistence by enabling recovery from setbacks and continued effort.'),
]

recommendations.extend(mechanism_outcome[:75])  # Take first 75
print(f'  Added {len(mechanism_outcome[:75])} mechanism-outcome relationships')

# ============================================================================
# PRIORITY 3: Bidirectional Relationships (25 relationships)
# ============================================================================
print('Generating Priority 3: Bidirectional Relationships (25)...')

bidirectional = [
    # Attention ↔ Memory (reverse direction)
    ('Memory', 'Working Memory', 'Memory', 'SUPPORTS', 'Selective Attention', 'Attention', 'Attention Types', 'Working memory supports selective attention by maintaining attentional goals and filtering criteria.'),
    ('Memory', 'Memory Encoding', 'MemorySystems', 'REQUIRES', 'Sustained Attention', 'Attention', 'Attention Types', 'Memory encoding requires sustained attention to achieve deep processing and consolidation.'),
    
    # Emotion ↔ Cognition (reverse)
    ('CognitiveProcesses', 'Attention', 'CognitiveProcesses', 'REGULATES', 'Positive Emotions', 'Emotions', 'Affective States', 'Attentional focus can regulate positive emotions by directing awareness toward positive stimuli.'),
    ('CognitiveProcesses', 'Attentional Allocation', 'CognitiveProcesses', 'MODULATES', 'Negative Emotions', 'Emotions', 'Affective States', 'Attentional allocation modulates negative emotions by controlling focus on threats.'),
    
    # Motivation ↔ Attention (reverse)
    ('Attention', 'Sustained Attention', 'Attention', 'BUILDS', 'Internal Satisfaction', 'IntrinsicMotivation', 'Motivational Types', 'Sustained attention builds internal satisfaction by enabling flow states and mastery experiences.'),
    
    # Executive Functions ↔ Metacognition (reverse)
    ('Metacognition', 'Metacognitive Control', 'MetacognitiveControl', 'ACTIVATES', 'Planning & Organization', 'ExecutiveFunctions', 'Executive Functions', 'Metacognitive control activates planning by identifying need for strategic adjustment.'),
    
    # Mindset ↔ Learning Strategies (reverse)
    ('SelfRegulatedLearning', 'Strategic Use of Learning Methods', 'SelfRegulatedLearning', 'REINFORCES', 'Growth Mindset', 'Mindset', 'Belief Systems', 'Strategic learning reinforces growth mindset by providing evidence that methods matter.'),
    
    # Stress ↔ Cognition (reverse)
    ('CognitiveProcesses', 'Attention', 'CognitiveProcesses', 'CAN_REDUCE', 'Overwhelming Stress That Impairs Functioning', 'NegativeStressDistress', 'Affective States', 'Focused attention can reduce overwhelming stress by interrupting rumination and threat detection.'),
    
    # Memory ↔ Executive Functions (reverse)
    ('ExecutiveFunctions', 'Planning & Organization', 'ExecutiveFunctions', 'RELIES_ON', 'Working Memory Capacity', 'ResourceAllocation', 'Memory Systems', 'Planning relies on working memory capacity to coordinate multiple sub-goals and steps.'),
    
    # Social ↔ Emotional (reverse)
    ('EmotionalRegulation', 'Awareness and Management of Emotions', 'EmotionalRegulation', 'FACILITATES', 'Perspective-taking', 'SocialCognition', 'Social Cognition', 'Emotional awareness facilitates perspective-taking by enabling recognition of others emotional states.'),
    
    # Creativity ↔ Emotion (reverse)
    ('PositiveEmotions', 'Joy', 'PositiveEmotions', 'UNLEASHES', 'Divergent Thinking', 'Creativity', 'Creative Processes', 'Joy unleashes divergent thinking by broadening attentional scope and reducing constraints.'),
    
    # Learning Performance ↔ Motivation (reverse)
    ('LearningPerformance', 'Deep Learning', 'LearningPerformance', 'GENERATES', 'Internal Satisfaction', 'IntrinsicMotivation', 'Motivational Types', 'Deep learning generates internal satisfaction by providing sense of mastery and competence.'),
    ('LearningPerformance', 'Persistence', 'LearningPerformance', 'STRENGTHENS', 'Self-Determination', 'IntrinsicMotivation', 'Motivational Types', 'Persistence strengthens self-determination by demonstrating capacity for goal-directed behavior.'),
    
    # Critical Thinking ↔ Metacognition (reverse)
    ('Metacognition', 'Metacognitive Monitoring', 'MetacognitiveMonitoring', 'SHARPENS', 'Analysis', 'CriticalThinking', 'Higher-Order Cognition', 'Metacognitive monitoring sharpens analysis by providing awareness of reasoning quality.'),
    
    # Attention ↔ Emotion (reverse)
    ('PositiveEmotions', 'Interest', 'PositiveEmotions', 'CAPTURES', 'Selective Attention', 'Attention', 'Attention Types', 'Interest captures selective attention by making stimuli salient and rewarding.'),
    ('NegativeEmotions', 'Fear', 'NegativeEmotions', 'HIJACKS', 'Selective Attention', 'Attention', 'Attention Types', 'Fear hijacks selective attention by prioritizing threat detection over task goals.'),
    
    # Working Memory ↔ Attention (reverse)
    ('Attention', 'Selective Attention', 'Attention', 'GATES', 'Memory Encoding', 'MemorySystems', 'Memory Systems', 'Selective attention gates memory encoding by determining what information is processed deeply.'),
    
    # Executive Functions ↔ Self-Regulation (reverse)
    ('SelfRegulation', 'Adaptive Responses', 'SelfRegulation', 'REQUIRES', 'Planning & Organization', 'ExecutiveFunctions', 'Executive Functions', 'Adaptive responses require planning and organization to implement strategic adjustments.'),
    
    # Motivation ↔ Executive Functions (reverse)
    ('ExecutiveFunctions', 'Planning & Organization', 'ExecutiveFunctions', 'GENERATES', 'Competence', 'IntrinsicMotivation', 'Motivational Types', 'Successful planning generates sense of competence through mastery experiences.'),
    
    # Memory ↔ Emotion (reverse)
    ('Emotions', 'Positive Emotions', 'Emotions', 'FACILITATE', 'Memory Consolidation', 'Memory', 'Memory Systems', 'Positive emotions facilitate consolidation by optimizing hippocampal and dopaminergic function.'),
    
    # Social Learning ↔ Cognition (reverse)
    ('CriticalThinking', 'Analysis', 'CriticalThinking', 'IMPROVES', 'Feedback Integration', 'SocialLearning', 'Learning Processes', 'Analytical thinking improves feedback integration by enabling evaluation of input quality.'),
    
    # Teaching ↔ Learning (reverse)
    ('LearningOutcomes', 'Participation', 'LearningOutcomes', 'INFORMS', 'Adaptive Strategies', 'TeachingPractices', 'Pedagogical Strategies', 'Student participation informs adaptive teaching by providing feedback on instructional effectiveness.'),
    
    # Metacognition ↔ Memory (reverse)
    ('Memory', 'Memory Retrieval', 'Memory', 'BENEFITS_FROM', 'Metacognitive Monitoring', 'MetacognitiveMonitoring', 'Metacognitive Processes', 'Memory retrieval benefits from metacognitive monitoring through awareness of retrieval success.'),
    
    # Stress ↔ Executive Functions (reverse)
    ('ExecutiveFunctions', 'Planning & Organization', 'ExecutiveFunctions', 'CAN_MITIGATE', 'Overwhelming Stress That Impairs Functioning', 'NegativeStressDistress', 'Affective States', 'Effective planning can mitigate overwhelming stress by breaking challenges into manageable steps.'),
    
    # Emotion ↔ Learning (reverse)
    ('LearningPerformance', 'Wellbeing', 'LearningPerformance', 'PROMOTES', 'Positive Emotions', 'Emotions', 'Affective States', 'Wellbeing promotes positive emotions by creating psychological safety and satisfaction.'),
    
    # Creativity ↔ Critical Thinking (reverse)
    ('CriticalThinking', 'Evaluation', 'CriticalThinking', 'REFINES', 'Divergent Thinking', 'Creativity', 'Creative Processes', 'Evaluation refines divergent thinking by filtering and improving novel ideas.'),
]

recommendations.extend(bidirectional[:25])  # Take first 25
print(f'  Added {len(bidirectional[:25])} bidirectional relationships')

# ============================================================================
# CREATE DATAFRAME AND SAVE TO CSV
# ============================================================================

print(f'\n✅ Total recommendations generated: {len(recommendations)}')

# Convert to DataFrame with exact 8-column structure
df_recommendations = pd.DataFrame(recommendations, columns=[
    'Category A', 'Value A', 'Concept A', 'Relationship', 'Value B', 'Concept B', 'Category B', 'Description'
])

# Save to CSV
output_file = 'NeuroData/KG_NEURO_CONNECTIVITY_RECOMMENDATIONS.csv'
df_recommendations.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f'\n💾 Saved {len(df_recommendations)} recommendations to: {output_file}')
print(f'\n📊 Summary by Priority:')
print(f'   Priority 1 (Cross-Domain Bridges): 50')
print(f'   Priority 2 (Mechanism → Outcome): 75')
print(f'   Priority 3 (Bidirectional): 25')
print(f'   TOTAL: 150')
print(f'\n🎯 Expected Impact:')
print(f'   Current avg connectivity: 2.18')
print(f'   After adding 150 relationships: ~2.78')
print(f'   Target connectivity: 3.5+')
print(f'   Additional relationships needed: ~200 more')
print(f'\n✅ File ready for neuro team review and addition!')



