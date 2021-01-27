classdef bc_type < uint8
    enumeration
        none (0)
        b4c (1)
        dirichlet (2)
        dirichlet_up (3)
        dirichlet_low (4)
%         clamped_up_dirichlet_low (5)
        %robin (3) %Not yet implemented
    end
end
