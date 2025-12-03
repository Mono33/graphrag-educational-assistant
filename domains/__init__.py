#!/usr/bin/env python3
"""
Domain Registry
Auto-discover and load domain configurations

This module provides centralized access to all domain configurations.
Adding a new domain is as simple as:
1. Create a new domain config file (e.g., math_domain.py)
2. Register it in this file
3. Done! All components automatically use it.
"""

from typing import Dict, Optional, List
from domains.base_config import BaseDomainConfig


class DomainRegistry:
    """Registry for all domain configurations"""
    
    def __init__(self):
        """Initialize domain registry"""
        self._domains: Dict[str, BaseDomainConfig] = {}
        self._register_domains()
    
    def _register_domains(self):
        """Register all available domains
        
        To add a new domain:
        1. Import the domain config class
        2. Instantiate it
        3. Register it in self._domains
        """
        # Import domain configurations
        from domains.udl_domain import UDLDomainConfig
        from domains.neuro_domain import NeuroDomainConfig
        
        # Register UDL domain
        udl = UDLDomainConfig()
        self._domains[udl.name] = udl
        print(f"[OK] Registered domain: {udl.name} ({udl.display_name})")
        
        # Register Neuro domain
        neuro = NeuroDomainConfig()
        self._domains[neuro.name] = neuro
        print(f"[OK] Registered domain: {neuro.name} ({neuro.display_name})")
        
        # ============================================================
        # FUTURE DOMAINS: Just add here!
        # ============================================================
        # Example:
        # from domains.math_domain import MathDomainConfig
        # math = MathDomainConfig()
        # self._domains[math.name] = math
        # print(f"✅ Registered domain: {math.name} ({math.display_name})")
    
    def get(self, domain_name: str) -> Optional[BaseDomainConfig]:
        """Get domain configuration by name
        
        Args:
            domain_name: Domain identifier ('udl', 'neuro', 'math', etc.)
            
        Returns:
            Domain configuration or None if not found
            
        Note:
            Returns None for domain_name="all" (means merge all domains)
        """
        if domain_name == "all":
            return None  # "all" is special case (merge multiple domains)
        
        return self._domains.get(domain_name)
    
    def list_all(self) -> List[str]:
        """List all registered domain names
        
        Returns:
            List of domain identifiers (e.g., ['udl', 'neuro'])
        """
        return list(self._domains.keys())
    
    def get_all_configs(self) -> List[BaseDomainConfig]:
        """Get all domain configurations
        
        Returns:
            List of domain configuration instances
        """
        return list(self._domains.values())
    
    def validate_all(self) -> Dict[str, tuple]:
        """Validate all registered domains
        
        Returns:
            Dict mapping domain_name -> (is_valid, errors)
        """
        results = {}
        for name, domain_config in self._domains.items():
            is_valid, errors = domain_config.validate()
            results[name] = (is_valid, errors)
        
        return results
    
    def __repr__(self) -> str:
        """String representation"""
        return f"<DomainRegistry: {len(self._domains)} domains registered ({', '.join(self._domains.keys())})>"


# ============================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================

# Create global registry (lazy initialization on first import)
_registry = None

def get_registry() -> DomainRegistry:
    """Get global domain registry (singleton)
    
    Returns:
        Domain registry instance
    """
    global _registry
    if _registry is None:
        _registry = DomainRegistry()
    return _registry


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_domain_config(domain: str) -> Optional[BaseDomainConfig]:
    """Get domain configuration (convenience function)
    
    Args:
        domain: Domain identifier ('udl', 'neuro', 'all', etc.)
        
    Returns:
        Domain configuration or None if not found or domain="all"
        
    Example:
        >>> config = get_domain_config("udl")
        >>> weights = config.get_node2vec_weights()
        >>> print(weights['StudentWithSpecialNeeds'])  # 3.0
    """
    registry = get_registry()
    return registry.get(domain)


def list_available_domains() -> List[str]:
    """List all available domain identifiers
    
    Returns:
        List of domain names (e.g., ['udl', 'neuro'])
        
    Example:
        >>> domains = list_available_domains()
        >>> print(domains)  # ['udl', 'neuro']
    """
    registry = get_registry()
    return registry.list_all()


def validate_domains() -> Dict[str, tuple]:
    """Validate all registered domains
    
    Returns:
        Dict mapping domain_name -> (is_valid, errors)
        
    Example:
        >>> results = validate_domains()
        >>> for domain, (valid, errors) in results.items():
        ...     if not valid:
        ...         print(f"{domain}: {errors}")
    """
    registry = get_registry()
    return registry.validate_all()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    'DomainRegistry',
    'get_registry',
    'get_domain_config',
    'list_available_domains',
    'validate_domains'
]

